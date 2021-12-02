Client Bootstrapping Notes
------------------------
All NDN entities requires security bootstrapping to function properly.
Security bootstrapping gives an NDN entity with: **name, trust anchor, certificate, trust policies**.
This note briefly documents the python-based implementation icear client boostrapping.

## Before bootstrapping
Server's certificate chain:
```
/ndn/edu/ucla/icear
+-> /ndn/KEY/%EC%F1L%8EQ%23%15%E0/ndn/%FD%00%00%01u%E6%7F2%10 (testbed root certificate, trust anchor)
    +-> /ndn/edu/ucla/KEY/%87%BB%D8%99%AFA%BD%EC/NA/%FD%00%00%01vv%B6M%02 (ucla site certificate)
        +-> /ndn/edu/ucla/icear/KEY/%3E%F7%A5%5B%A2%16%F4h/NA/54=%00%00%01%7Ds%CB%9F%87 (icear server certificate)
```

Client's certificate chain:
```
/ndn/edu/ucla/cs/tianyuan-icear-client
+-> /ndn/KEY/%EC%F1L%8EQ%23%15%E0/ndn/%FD%00%00%01u%E6%7F2%10 (testbed root certificate, trust anchor)
    +-> /ndn/edu/ucla/KEY/%87%BB%D8%99%AFA%BD%EC/NA/%FD%00%00%01vv%B6M%02 (ucla site certificate)
        +-> /ndn/edu/ucla/cs/tianyuan-icear-client/KEY/%D5%3F%91%91%E4u4%A3/NA/v=1638271730898 (client certificate)
```

Client wants to self-assign a name ``/ndn/edu/ucla/icear/tianyuan-icear-client`` apply a certificate under ``/ndn/edu/ucla/icear``.

## After bootstrapping
Server's certificate chain:
```
/ndn/edu/ucla/icear
+-> /ndn/KEY/%EC%F1L%8EQ%23%15%E0/ndn/%FD%00%00%01u%E6%7F2%10 (testbed root certificate, trust anchor)
    +-> /ndn/edu/ucla/KEY/%87%BB%D8%99%AFA%BD%EC/NA/%FD%00%00%01vv%B6M%02 (ucla site certificate)
        +-> /ndn/edu/ucla/icear/KEY/%3E%F7%A5%5B%A2%16%F4h/NA/54=%00%00%01%7Ds%CB%9F%87 (icear server certificate)
```

Client's certificate chain:
```
/ndn/edu/ucla/cs/tianyuan-icear-client (identity 1, not used for communication with server)
+-> /ndn/KEY/%EC%F1L%8EQ%23%15%E0/ndn/%FD%00%00%01u%E6%7F2%10 (testbed root certificate, trust anchor)
    +-> /ndn/edu/ucla/KEY/%87%BB%D8%99%AFA%BD%EC/NA/%FD%00%00%01vv%B6M%02 (ucla site certificate)
        +-> /ndn/edu/ucla/cs/tianyuan-icear-client/KEY/%D5%3F%91%91%E4u4%A3/NA/v=1638271730898 (client certificate 1)

/ndn/edu/ucla/icear/tianyuan-icear-client (identity 2)
+-> /ndn/edu/ucla/icear/KEY/%3E%F7%A5%5B%A2%16%F4h/self/%FD%00%00%01%7Ds%C6%CB%FD (icear server self-signed certificate, local trust anchor)
    +->  /ndn/edu/ucla/icear/tianyuan-icear-client/KEY/%9AY%C3%9E%F4%BF%DCx/icear-server/v=1638466685273 (client certificate 2)
            (actual key id and version number depend)
```

## Client Bootstrapping Procedures

### Step 1: Mutual authentication
* **Step 1.A:** Client authenticates server by accepting server's self-signed certificate from out-of-band channel (i.e., git clone this repo).
* **Step 1.B:** Server authenticates client by validating client's testbed-signed certificate, with the testbed root certificate as certificate chain trust anchor.

### Step 2: Installing security components  
* **Step 2.A:** Client installs trust anchor (completed in Step 1.A)
* **Step 2.B:** Client self-assigns a name by concatenating the trust anchor prefix and partial suffix of client's testbed-signed certificate and get ``/ndn/edu/ucla/icear/tianyuan-icear-client``
* **Step 2.C:** Client applies a certificate for ``/ndn/edu/ucla/icear/tianyuan-icear-client``
* **Step 2.D:** Client obtains trust policy under ``/ndn/edu/ucla/icear`` (TODO later, no real implementation yet)

## Usage

### Prerequisite
ndn-cxx, NFD, **latest python-ndn from source**
You can install latest python-ndn by:
```
pip3 install -U git+https://github.com/named-data/python-ndn.git
```

### Server:
```
ndn-autoconfig (this connect you to the NDN testbed, so you can verify testbed-signed certificates)
python3 test-server.py
```

### Client:
```
(Make sure the client-facing NFD knows how to reach /edge. Can run both server and client on one machine)
python3 test-client.py
```

## Implementation Notes
### Certificate issuance:
One important goal of this python-based bootstrapping demo is to guide the later android-based client implementation.
It is tricky, and demands more workload to adapt C++ based [NDNCERT](https://github.com/named-data/ndncert) library on those android clients.
Therefore we decided to implement similar packet exchanges of NDNCERT [Proof-of-Possession Challenge](https://github.com/named-data/ndncert/wiki/NDNCERT-Protocol-0.3-Challenges) in jndn later.
This python-ndn based implementation gives the sense of how later jndn-based would be structured.
This is not a general solution, and not as secured as NDNCERT.

### Server bootstrapping:
Server is bootstrapped in ``bootstrap()`` of ``test-server.py``. 
It gives testbed root certificate as trust anchor.

### Safebag:
This implementation uses hard-coded safebags to directly import NDN testbed certified key pairs (including the private keys) for a quick demo, so please do not propagate.