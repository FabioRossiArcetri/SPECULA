import unittest

import specula
specula.init(0)  # Default target device

from specula import cp, np, cpuArray
from specula.base_processing_obj import BaseProcessingObj
from specula.base_value import BaseValue
from specula.connections import InputValue
from specula.loop_control import LoopControl
from iteras.processing_objects.wave_generator import WaveGenerator
from iteras.data_objects.merger import Merger as MergerOptions
from iteras.merger import Merger


class Doubler(BaseProcessingObj):
    '''out = 2 * in, with options to break the rules of merged objects'''

    def __init__(self, target_device_idx=None, realloc=False, cublas=False, mergeable=True,
                 time_in_trigger=False):
        super().__init__(target_device_idx=target_device_idx)
        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.out = BaseValue(value=self.xp.zeros(4, dtype=self.dtype), target_device_idx=target_device_idx)
        self.outputs['out_value'] = self.out
        self.buf = self.xp.zeros(4, dtype=self.dtype)
        self.mat = self.xp.eye(4, dtype=self.dtype)
        self.realloc = realloc
        self.cublas = cublas
        self.mergeable = mergeable
        self.time_in_trigger = time_in_trigger

    def can_merge(self):
        return self.mergeable

    def prepare_trigger(self, t):
        super().prepare_trigger(t)
        v = self.local_inputs['in_value'].value
        if self.realloc:
            self.buf = v * 1.0      # a new array at each step
        else:
            self.buf[:] = v

    def trigger_code(self):
        if self.cublas:
            self.out.value[:] = 2 * (self.mat @ self.buf)
        else:
            self.out.value[:] = 2 * self.buf
        if self.time_in_trigger:
            self.out.generation_time = self.current_time

    def post_trigger(self):
        super().post_trigger()
        if not self.time_in_trigger:
            self.out.generation_time = self.current_time


class Counter(BaseProcessingObj):
    '''Counts the steps in which its input was refreshed'''

    def __init__(self, target_device_idx=None):
        super().__init__(target_device_idx=target_device_idx)
        self.inputs['in_value'] = InputValue(type=BaseValue)
        self.count = 0

    def trigger_code(self):
        self.count += 1


@unittest.skipIf(cp is None, 'Cupy not found')
class TestMerger(unittest.TestCase):

    def run_chain(self, merge, steps=12, options=None, on_step=None, **kwargs):
        '''Generator -> Doubler -> Counter. Return the Doubler outputs, the objects and the merger'''
        gen = WaveGenerator(amp=[1.0], freq=[37.0], constant=[0.5], output_size=4, target_device_idx=0)
        dbl = Doubler(target_device_idx=0, **kwargs)
        cnt = Counter(target_device_idx=0)
        dbl.inputs['in_value'].set(gen.outputs['output'])
        cnt.inputs['in_value'].set(dbl.outputs['out_value'])
        loop = LoopControl()
        for i, obj in enumerate([gen, dbl, cnt]):
            loop.add(obj, i)
        merger = None
        if merge:
            merger = Merger(options or MergerOptions())
            merger.apply([gen, dbl, cnt])
        loop.start(run_time=steps * 1e-3, dt=1e-3)
        out = []
        for i in range(steps):
            loop.iter()
            if on_step is not None:
                on_step(i, gen, dbl)
            out.append(cpuArray(dbl.out.value).copy())
        loop.finish()
        return np.array(out), (gen, dbl, cnt), merger

    def check_same(self, merge_kwargs=None, **kwargs):
        ref, _, _ = self.run_chain(False, **kwargs)
        out, objs, merger = self.run_chain(True, **(merge_kwargs or {}), **kwargs)
        np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)
        # The generator depends on the time: the outputs change at each step
        self.assertTrue(np.all(np.abs(np.diff(ref[:, 0])) > 1e-4))
        return objs, merger

    def status(self, merger, name):
        return {obj.name: runner.status for obj, runner, _ in merger.rows if runner is not None}[name]

    def test_merged_chain_matches_normal_run(self):
        (gen, dbl, cnt), merger = self.check_same()
        self.assertEqual(self.status(merger, 'WaveGenerator'), 'merged')
        self.assertEqual(self.status(merger, 'Doubler'), 'merged')
        self.assertIsNotNone(dbl._merge.graph)
        # The counter has no graph-safe trigger_code: not merged, but triggered at each step
        self.assertTrue(self.status(merger, 'Counter').startswith('can_merge()'))
        self.assertEqual(cnt.count, 12)

    def test_reallocated_array_stops_merging(self):
        with self.assertLogs('iteras.merger', level='WARNING') as logs:
            (_, dbl, _), merger = self.check_same(realloc=True)
        self.assertIsNone(dbl._merge)
        self.assertIn('array buf reallocated', self.status(merger, 'Doubler'))
        self.assertIn('not merged anymore', logs.output[0])

    def test_failed_capture_stops_merging(self):
        with self.assertLogs('iteras.merger', level='WARNING'):
            (_, dbl, _), merger = self.check_same(cublas=True)
        self.assertIsNone(dbl._merge)
        self.assertIn('capture failed', self.status(merger, 'Doubler'))

    def test_can_merge_false(self):
        (_, dbl, _), merger = self.check_same(mergeable=False)
        self.assertIsNone(dbl._merge.graph)
        self.assertEqual(self.status(merger, 'Doubler'), 'can_merge(): False')

    def test_output_time_set_in_trigger_code(self):
        (_, _, cnt), merger = self.check_same(time_in_trigger=True)
        self.assertEqual(self.status(merger, 'Doubler'), 'merged')
        # The consumer sees the output refreshed at each step, as without merger
        self.assertEqual(cnt.count, 12)

    def test_invalidate_captures_again(self):
        graphs = []

        def on_step(i, gen, dbl):
            if dbl._merge is not None:
                graphs.append(dbl._merge.graph)
            if i == 5:
                dbl.invalidate_graph()

        (_, dbl, _), merger = self.check_same(merge_kwargs={'on_step': on_step})
        self.assertEqual(self.status(merger, 'Doubler'), 'merged')
        # Graph recorded at each step, before the invalidation of step 5
        self.assertIs(graphs[4], graphs[5])
        self.assertIsNot(graphs[5], graphs[6])
        self.assertIs(graphs[6], graphs[7])

    def test_capture_after(self):
        def on_step(i, gen, dbl):
            # Captured at the trigger after capture_after triggers
            self.assertEqual(dbl._merge.graph is not None, i >= 3)

        self.check_same(merge_kwargs={'on_step': on_step,
                                      'options': MergerOptions(capture_after=3)})

    def test_excluded_object(self):
        (_, dbl, _), merger = self.check_same(merge_kwargs={'options': MergerOptions(exclude=['Doubler'])})
        self.assertIsNone(dbl._merge)
        reasons = {obj.name: reason for obj, _, reason in merger.rows}
        self.assertEqual(reasons['Doubler'], 'excluded')
        self.assertIn('Doubler', merger.report())
        self.assertIn('1 of 3 objects merged', merger.report())

    def test_merged_object_does_not_wait_for_the_gpu(self):
        (_, dbl, _), _ = self.check_same()
        self.assertIsNotNone(dbl._merge)
        # post_trigger() of a merged object with its own graph does not wait for its stream
        dbl.stream = cp.cuda.Stream()
        dbl.cuda_graph = object()
        dbl.inputs_changed = True
        dbl.post_trigger()


class TestMergerOptions(unittest.TestCase):

    def test_capture_after_must_be_positive(self):
        with self.assertRaises(ValueError):
            MergerOptions(capture_after=0)

    def test_simul_sections(self):
        from specula.simul import Simul
        sim = Simul('unused.yml')
        params = {'main': {'class': 'SimulParams'}, 'merger': {'class': 'Merger', 'exclude': ['a']}}
        self.assertEqual(sim.merger_keys(params), ['merger'])
        options = sim.take_merger_section(params)
        self.assertEqual(options.exclude, ['a'])
        self.assertNotIn('merger', params)
        self.assertIsNone(sim.take_merger_section(params))
        sim.add_merger_section(params)
        self.assertEqual(params['merger'], {'class': 'Merger'})
        # Not added twice
        sim.add_merger_section(params)
        self.assertEqual(sim.merger_keys(params), ['merger'])

    def test_allocation_of_other_merge_mode_is_stale(self):
        from iteras.allocator import check_meta, StaleAllocation
        meta = {'signature': {}, 'pins': {}, 'architecture': {}, 'precision': 1, 'merge': False}
        check_meta(dict(meta), dict(meta), 'f')
        # Allocations written before the merger existed were computed without it
        check_meta({k: v for k, v in meta.items() if k != 'merge'}, dict(meta), 'f')
        with self.assertRaisesRegex(StaleAllocation, 'merger enabled'):
            check_meta(dict(meta), dict(meta, merge=True), 'f')


if __name__ == '__main__':
    unittest.main()
