import io
import yaml

P = 'config/ANDES/andes_pl_piston_pyr.yml'
s = io.open(P, encoding='utf-8').read()
old = ("  sign:              1                      # the convention the interaction matrix used\n"
       "  height:            0")
new = ("  sign:              1                      # the convention the interaction matrix used\n"
       "  nmodes:            500                    # exactly the calibrated modes of the pyramid loop\n"
       "  height:            0")
if 'nmodes:            500' in s and 'height:            0\n  inputs:\n    in_command:      \'control_ho' in s:
    print('already patched')
elif old in s:
    s = s.replace(old, new, 1)
    io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
    print('dm_ho.nmodes added')
else:
    raise SystemExit('anchor not found')

d = yaml.safe_load(io.open(P, encoding='utf-8').read())
print('dm_ho:', {k: d['dm_ho'][k] for k in ('nmodes', 'sign', 'ifunc_ref') if k in d['dm_ho']})
print('control_ho n_modes:', d['control_ho']['n_modes'])
