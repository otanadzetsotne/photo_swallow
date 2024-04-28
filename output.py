import os


def get_out_dir():
    if not hasattr(get_out_dir, 'out'):
        dir_ = os.path.dirname(__file__)
        out = os.path.join(dir_, '.output')
        os.makedirs(out, exist_ok=True)
        print(f'Output directory: {out}')
        get_out_dir.out = out

    return get_out_dir.out


def output(path=''):
    out = get_out_dir()

    if path:
        out = os.path.join(out, path)
        os.makedirs(os.path.dirname(out), exist_ok=True)

    return out


if __name__ == '__main__':
    print(output('dir1/dir2/test.jpg'))
