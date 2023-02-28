git submodule update --init
pip install -r ./requirements.txt
cd mega || exit
pip install -e .
cd ..
cd ps4-rs || exit
maturin build --release
pip install --force-reinstall target/wheels/*.whl