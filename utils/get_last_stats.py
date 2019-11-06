import re
import os

def get_last_stats(output_dir, epo=None):
    p = re.compile(r'(.+)_epo([+-]?\d+).pth')
    files = [[f, p.match(os.path.basename(f)).groups()[0], int(p.match(os.path.basename(f)).groups()[1])]
             for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f)) and p.match(f) is not None]

    if len(files)==0: return {}

    if epo is None:
        files.sort(key=lambda t: t[2], reverse=True)
        last_epo = files[0][2]
        checkpoints = {'epo': last_epo}
        for file in files:
            if file[2] != last_epo:
                break
            checkpoints[file[1]] = os.path.join(output_dir, file[0])

        return checkpoints
    else:
        checkpoints = {'epo': epo}
        for file in files:
            if file[2] != epo:
                continue
            checkpoints[file[1]] = os.path.join(output_dir, file[0])
        return checkpoints

if __name__ == '__main__':
    print(get_last_stats('/home/tangyingtian/Person_ReID_Cross_Domain/checkpoint/Cross_domain/baseline_tuned/', 80))
