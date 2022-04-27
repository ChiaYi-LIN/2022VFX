rm -rf ./hw2_[46]
mkdir -p ./hw2_[46]

cp -r ./data ./hw2_[46]
cp -r ./code ./hw2_[46]
cp ./README.md ./hw2_[46]
cp ./report.pdf ./hw2_[46]
cp ./result.png ./hw2_[46]

rm -f hw2_[46].zip
zip -r hw2_[46].zip ./hw2_[46]

rm -rf ./hw2_[46]