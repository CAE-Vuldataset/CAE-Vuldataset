git submodule init
git submodule update
mkdir build
mkdir -p build/tensoradapter/pytorch

CACHEDIR=/tmp/dgl_binaries

if [[ "$OSTYPE" = "darwin"* ]]; then
	echo Copying prebuilt CPU binary...
	cp /tmp/libdgl.dylib build
else
	if [[ $USE_CUDA = 'ON' ]]; then
		echo Copying prebuilt CUDA $CUDA_VER binary...
		cp -v $CACHEDIR/libdgl.so.cu$CUDA_VER build/libdgl.so
		cp -v $CACHEDIR/tensoradapter_pytorch_cu$CUDA_VER/*.so build/tensoradapter/pytorch
	else
		echo Copying prebuilt CPU binary...
		cp -v $CACHEDIR/libdgl.so.cpu build/libdgl.so
		cp -v $CACHEDIR/tensoradapter_pytorch_cpu/*.so build/tensoradapter/pytorch
	fi
fi
cd python
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
