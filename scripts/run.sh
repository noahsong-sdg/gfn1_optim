#!/usr/bin/env bash

# Uncomment for debugging
# set -ex

# Input from fit runner with defaults for standalone use
fitpar=$(realpath ${TBLITE_PAR:-./si_fitpar.toml})
output=$(realpath ${TBLITE_OUT:-.data})
tblite=$(which ${TBLITE_EXE:-tblite})

# Balance to get nproc == njob * nthreads
nthreads=4
njob=$(nproc | awk "{print int(\$1/$nthreads)}")

# Temporary data file
data=.data

# Temporary wrapper
wrapper=./wrapped_runner

# Arguments for tblite runner
tblite_args="run --param \"$fitpar\" --grad results.tag coord"

# Ad hoc error in case the Hamiltonian does not work
# (SCC does not converge or similar)
penalty="1.0e3"

# Create our wrapper script
cat > "$wrapper" <<-EOF
#!/usr/bin/env bash
if [ -d \$1 ]; then
  pushd "\$1" > /dev/null 2>&1
  test -f "$data" && rm "$data"
  OMP_NUM_THREADS=1 "$tblite" $tblite_args  > tblite.out 2> tblite.err \
    || echo "0.0 $penalty  # run: \$1" > "$data"
  "$tblite" tagdiff --fit results.tag reference.tag >> "$data" \
    || echo "0.0 $penalty  # diff: \$1" >> "$data"
fi
EOF
chmod +x "$wrapper"

# Create the actual multiprocessing queue
printf "%s\0" data/*/ | xargs -n 1 -P $njob -0 "$wrapper"

# Collect the data
cat data/*/$data > "$output"

# Cleanup
rm data/*/$data
rm "$wrapper"
