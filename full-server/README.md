# NEAR Edge Server

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

Config the producers' routing with NFDC. For example:
```
 nfdc route add /NEAR/group1/producer1 udp4://192.168.1.100:6363
```

To run the NEAR edge, start the server with 
```
 python ndn-server-multi-ar-task.py
```
The server will display the processed frame. Then open the consumer app & click "edgeU" or "edgeM" to retrieve AR contents with unicast or multicast.

## License

NEAR is a free and open-source platform licensed under the GPL v3. For more information about licensing, refer to
[`LICENSE`](LICENSE).