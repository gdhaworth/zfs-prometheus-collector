# ZFS Prometheus Collector

A quick and dirty Prometheus collector for ZFS pools and datasets in Python. It is not quite finished; for example it
doesn't handle snapshots or allow customization via command line arguments. Seems to work so far though!


### Installation

I developed this using Python 3.12, but it may work just fine with earlier versions. Set up your Python environment
(`pyenv`, `virtualenv`, whatever) and:

```shell
pip install -r requirements.txt
python collector.py
```
