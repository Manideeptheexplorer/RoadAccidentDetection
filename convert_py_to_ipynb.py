import jupytext
import sys

def convert_py_to_ipynb(py_filename):
    # Read the Python script as a notebook using Jupytext
    notebook = jupytext.read(py_filename)
    # Create the output filename by replacing .py with .ipynb
    ipynb_filename = py_filename.rsplit('.', 1)[0] + '.ipynb'
    # Write the notebook object to an ipynb file
    jupytext.write(notebook, ipynb_filename)
    print(f"Converted {py_filename} to {ipynb_filename}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_py_to_ipynb.py your_script.py")
    else:
        py_filename = sys.argv[1]
        convert_py_to_ipynb(py_filename)
