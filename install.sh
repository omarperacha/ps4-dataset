git submodule update --init
pip install -r ./requirements.txt
cd ps4-rs || exit
maturin build --release
pip install target/wheels/*.whl