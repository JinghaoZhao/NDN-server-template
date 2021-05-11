# NDN AR Server Template

## Install the ffmpeg and python libs

```
sudo snap install ffmpeg
pip3 install -r ./requirements.txt
```

## Install the customized NDN-libs
```
git clone https://github.com/JinghaoZhao/ndn-cxx.git
git clone https://github.com/JinghaoZhao/WING-NFD.git
```

Then compile and install the ndn-cxx and NFD with tutorials:
https://github.com/JinghaoZhao/ndn-cxx/blob/ndnar/docs/INSTALL.rst
https://github.com/JinghaoZhao/WING-NFD/blob/master/docs/INSTALL.rst

## Usage
Before running the server, start the NFD with 
```
 nfd-start
```
The current template using a local file as the input for testing convenience. To run the demo, first start the server
```
 python ndn-server-template.py
```
The server will display the processed frame. Then open the client with:

```
python NDNclient-testing.py
```
After the client getting the contents, it will also display frames.

## Development
1. Replace corresponding parts (line 145-161) with ML inference in the ndn-server-template. The related parts are highlighted with TODO.
2. Testing the server with the client to check if it can successfullly receive the processed stream.
