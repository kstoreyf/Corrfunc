#! /bin/bash

if ! python -c 'import numpydoc'; then easy_install --user numpydoc; fi
if ! python -c 'import sphinx'; then easy_install --user sphinx; fi

outdir=source/api
#sphinx-apidoc -H "Suave API reference" -M -f -o "$outdir" ../Corrfunc/ ../Corrfunc/tests.py ../Corrfunc/call_correlation_functions.py ../Corrfunc/call_correlation_functions_mocks.py
# the files after ../Corrfunc/ are excluded from doc generation
sphinx-apidoc -H "Suave API reference" -M -o "$outdir" ../Corrfunc/ ../Corrfunc/tests.py ../Corrfunc/call_correlation_functions.py ../Corrfunc/call_correlation_functions_mocks.py

tmpfile="$(mktemp)"
# Fix the blank sub-modules in the suave file
for docfile in "$outdir/suave.rst"
do
    # Delete three lines following the "submodules"
    sed -e '/Submodules/{N;N;d;}' "$docfile" > "$tmpfile"
    mv "$tmpfile"  "$docfile"
done

# Fix the duplicate entries for the various pair-counters
# (e.g., Corrfunc.theory.DD *and* Corrfunc.theory.DD.DD)
for docfile in "$outdir/suave.mocks.rst" "$outdir/suave.theory.rst"
do
    # Delete ALL lines following this "submodule" line in the theory/mocks
    # auto-generated documentation
    sed  -n '/Submodules/q;p' "$docfile" > "$tmpfile"
    mv "$tmpfile"  "$docfile"
done
