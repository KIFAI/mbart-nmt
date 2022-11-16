export REPO=${PWD%/*}
cp -pr $REPO/FastModel ./project
sudo rm -rf $PWD/project/FastModel/Onnx-MBart
cp -pr $REPO/Server ./project
cp -pr $REPO/WebApp ./project
