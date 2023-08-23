# SSL_brats_2class


uses ATLAS 
MICCAI ISLES 2022 challenge: https://atlas.grand-challenge.org/
MICCAI ISLES 2022 challenge provide a sample solution on GitHub to help you get started. https://github.com/npnl/isles_2022/#linux

decryption : openssl aes-256-cbc -md sha256 -d -a -in ATLAS_R2.0_encrypted.tar.gz -out ATLAS_R2.0.tar.gz <br/> 
and then write the password.
