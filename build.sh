rm -rf ./build
rm -f build.zip
docker build -t serverless-ml .
id=$(docker create serverless-ml)
docker cp $id:/usr/src/app/build ./build
docker rm -v $id
rm build/**/*.pyc
rm -rf build/**/test
rm -rf build/**/tests
cd build/ && zip -q -r -9 ../build.zip ./
rm -rf ./build
