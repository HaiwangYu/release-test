source /cvmfs/dune.opensciencegrid.org/products/dune/setup_dune.sh
setup dunesw v10_02_02d01 -q e26:prof

# source /exp/dune/app/users/yuhw/larsoft/v10_02_02d00/localProducts_larsoft_v10_02_02_e26_prof/setup
# mrbsetenv
# mrbslp

path-prepend /exp/dune/app/users/yuhw/opt/lib LD_LIBRARY_PATH

path-prepend /exp/dune/data/users/sergeym/PDHD_data_DNNROI/newFR/WireCellData/ WIRECELL_PATH
path-prepend /exp/dune/data/users/sergeym/PDHD_data_DNNROI/newFR/wire-cell-cfg/ WIRECELL_PATH
path-prepend /exp/dune/app/users/yuhw/release-test/h5tap/cfg WIRECELL_PATH
