import subprocess


def discover_test_runners_for(f: str):
    with open(f, 'r') as file:
        for line in file:
            if line.startswith('// RUN: '):
                yield line[8:].strip()

def run_tests_on(f: str):
    for runner in discover_test_runners_for(f):
        *test, check = runner.split("|")
        if '-check-prefix=' in check:
            check = check.split('-check-prefix=')[1].split(" ")[0].strip()
        elif '-check-prefix' in check:
            check = check.split('-check-prefix')[1].split(" ")[0].strip()
        else:
            check = 'CHECK'

        res = subprocess.check_output(
            ("|".join(test)).replace('%s', f"'{f}'"),
            shell=True,
            text=True,
        )

        yield check, res

def assemble_filecheck_lines(f: str):
    for check, res in run_tests_on(f):
        first = True
        for line in res.splitlines():
            if first and not line:
                continue
            if first:
                yield f'// {check}:      {line}'
                first = False
            else:
                if not line:
                    yield f"// {check}-EMPTY:"
                else:
                    yield f'// {check}-NEXT: {line}'
        yield ""

def main():
    import sys
    for f in sys.argv[1:]:
        print(f"\n\nTEST: {f}\n")
        for line in assemble_filecheck_lines(f):
            print(line)

if __name__ == '__main__':
    main()