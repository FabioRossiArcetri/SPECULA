import time

import matplotlib.pyplot as plt
from numbers import Integral

from specula.scalar_values import IntValue
from specula.base_processing_obj import BaseProcessingObj
from specula.base_processing_obj import OutputDesc

def runningOnNotebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None and 'IPKernelApp' in get_ipython().config
    except:
        return False

class BaseDisplay(BaseProcessingObj):

    __plot_completed = {}

    # Caps how often _safe_draw() actually repaints the GUI window (see
    # there for why a repaint is needed at all). Data updates
    # (_update_display) still happen on every trigger regardless -- this
    # only throttles the expensive part, so a display tapping a
    # fast-triggering input (e.g. every simulation step) doesn't spend a
    # large fraction of the run blocked on GUI repaints.
    _MIN_REDRAW_INTERVAL = 0.2  # [s] -> at most 5 repaints/second

    def __init__(self,
                 title='',
                 window: int=None,
                 subplot: int=111,
                 figsize=(8, 6)):
        super().__init__()

        if isinstance(window, Integral) and not isinstance(window, bool) and window >= 1:
            window = int(window)
            if window in self.__plot_completed.keys():
                raise ValueError(f'window {window} already exists')
        elif window is None:
            # Find an unused window number
            window = max(self.__plot_completed.keys(), default=0) + 1
        else:
            raise ValueError('window must be a positive integer')

        self.window = window
        self.figsize = figsize
        self.colorbar_added = False
        self.input_key = ''
        self.subplot = subplot
        self.onNotebook = runningOnNotebook()
        self._last_draw_time = 0.0

        if window not in self.__plot_completed:
            self.__plot_completed[window] = {}

        self.fig = plt.figure(num=self.window, figsize=self.figsize)
        self.ax = self.fig.add_subplot(self.subplot)
        self.__plot_completed[self.window][self.subplot] = False

        if title:
            self.ax.set_title(title)

        if not self.onNotebook:
            self.fig.show()
        else:
            from IPython.display import display
            self.handle = display(self.fig, display_id=True)

        self.output_id = IntValue(value=-1)
        self.outputs['out_window_id'] = self.output_id

    @classmethod
    def output_names(cls):
        return {'out_window_id': OutputDesc(IntValue, 'Window ID where the plot has been drawn')}

    def _update_display(self, data):
        """Update the display with new data"""
        raise NotImplementedError("Subclasses should implement this method")

    def _get_data(self):
        """Get data from input. Derived classes can override this method
        in case of complex data"""
        data = self.local_inputs.get(self.input_key)
        if data is None:
            self._show_error(f"No {self.input_key} data available")
            return
        return data

    def trigger_code(self):
        try:
            data = self._get_data()
            self._update_display(data)
            if self.onNotebook:
                self.handle.update(self.fig)
        except Exception as e:
            self._show_error(f"Display error: {str(e)}")

    def post_trigger(self):
        super().post_trigger()

        self.__plot_completed[self.window][self.subplot] = True
        self.output_id.value = self.window
        self.output_id.generation_time = self.current_time

        # If all subplots in this window have completed drawing,
        # call safe_draw() and reset the plot flags

        if all(self.__plot_completed[self.window].values()):
            self._safe_draw()
            for k in self.__plot_completed[self.window].keys():
                self.__plot_completed[self.window][k] = False

    # ============ UTILITY METHODS ============

    def _add_colorbar_if_needed(self, image_obj, unit=None, **kwargs):
        """Add colorbar if not already present"""
        if not hasattr(self, 'colorbar_added'):
            self.colorbar_added = False

        if not self.colorbar_added and image_obj is not None:
            cbar = plt.colorbar(image_obj, ax=self.ax, **kwargs)
            if unit:
                cbar.ax.set_title(unit)
            self.colorbar_added = True

    def _update_image_data(self, image_obj, data):
        """Standard image update logic"""
        if image_obj is not None:
            image_obj.set_data(data)
            # TODO: in some cases we want a fixed range,
            # so the clim should not be updated
            image_obj.set_clim(data.min(), data.max())

    def _show_error(self, message):
        self.ax.clear()
        self.ax.text(0.5, 0.5, message, ha='center', va='center',
                    transform=self.ax.transAxes, color='red', fontsize=12)
        self._safe_draw(force=True)

    # How long plt.pause() actually blocks to let the GUI event loop run.
    # A near-zero pause is enough to get a repaint through for a *local*
    # display, but not over X11 forwarding: that repaint is a real network
    # round-trip, and displays that only get triggered rarely (e.g. once
    # per training step, with several seconds of uninterrupted GPU work in
    # between, rather than every simulation step) don't get enough other
    # opportunities to compensate -- the paint commands just pile up
    # unflushed until the process exits. This needs to be long enough for
    # that round-trip regardless of how often _safe_draw() is called.
    _PAUSE_INTERVAL = 0.05  # [s]

    def _safe_draw(self, force=False):
        """Thread-safe drawing method, rate-limited to _MIN_REDRAW_INTERVAL
        (unless force=True, used for one-off events like _show_error).

        draw_idle()+flush_events() alone often isn't enough to force an
        actual on-screen repaint for a GUI backend (Tk/Qt/GTK) during a
        long-running, CPU-bound simulation loop that never otherwise
        yields to the GUI event loop -- the window can stay blank the
        whole run and only actually paint once, briefly, right as the
        process exits or is interrupted. plt.pause() explicitly pumps the
        event loop and forces the repaint.
        """
        now = time.monotonic()
        if not force and now - self._last_draw_time < self._MIN_REDRAW_INTERVAL:
            return
        self._last_draw_time = now
        try:
            if self.fig and self.fig.canvas:
                self.fig.canvas.draw_idle()
                plt.pause(self._PAUSE_INTERVAL)
        except Exception as e:
            self.logger.error(f"Drawing error: {e}")
