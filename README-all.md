# NEAR Platform

NEAR (NDN wirEless Augmented Reality) platform, an open-source AR platform with an information-centric design. 
![NEAR](./docs/NEAR.png)

## Install the ffmpeg and python libs on the edge

```
sudo snap install ffmpeg
pip3 install -r ./requirements.txt
```

## Install the customized NDN-libs on the edge
```
git clone https://github.com/JinghaoZhao/ndn-cxx.git
git clone https://github.com/JinghaoZhao/WING-NFD.git
```


Then compile and install the ndn-cxx and NFD with tutorials:

https://github.com/JinghaoZhao/ndn-cxx/blob/ndnar/docs/INSTALL.rst 

https://github.com/JinghaoZhao/WING-NFD/blob/master/docs/INSTALL.rst

## Usage
1. Before running the NEAR edge server, start the NFD on the edge server with: 
```
 nfd-start
```
2. Open the Android NFD & ARProducer App on the producer smartphone. Then click "Producer1" to start produce real-time camera view.

3. Config the producers' routing with NFDC. For example:
```
 nfdc route add /NEAR/group1/producer1 udp4://192.168.1.100:6363
```

4. To run the NEAR edge, start the server with the following command. The server will display the processed frame. 
```
 python ndn-server-multi-ar-task.py --task yolo
```

5. Then open the ARConsumer on the consumer device & click "edgeU" or "edgeM" to retrieve AR contents with unicast or multicast. The correct edge address need to be configured in the ARConsumer.

## Enable NDN-Wireless with link-layer mcast
If you want to leverage the high-performance link-layer multicast, the AP need to be [flashed with corresponding firmware](https://openwrt.org/docs/guide-user/installation/generic.flashing).
Now we support the TP-Link N750 Router and the corresponding firmware is in the "AP" folder.

## License

NEAR is a free and open-source platform licensed under the GPL v3. For more information about licensing, refer to
[`LICENSE`](LICENSE).