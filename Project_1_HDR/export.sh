# create directory
rm -rf ./hw1_[46]
mkdir -p ./hw1_[46]

# for data
mkdir -p ./hw1_[46]/data
mkdir -p ./hw1_[46]/data/original
cp ./data/* ./hw1_[46]/data
cp ./data/original/* ./hw1_[46]/data/original

# for code
mkdir -p ./hw1_[46]/code
cp ./code/* ./hw1_[46]/code
cp environment.yaml ./hw1_[46]/code

# readme
cp ./README.md ./hw1_[46]

# report
cp ./report.pdf ./hw1_[46]

# result
cp ./result.png ./hw1_[46]

# compress
rm -f hw1_[46].zip
zip -r hw1_[46].zip ./hw1_[46]

# clean directory
rm -r ./hw1_[46]

