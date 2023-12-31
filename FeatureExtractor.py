#Check if cython code has been compiled
import os
import pickle
import subprocess
print("Importing AfterImage Cython Library")
if not os.path.isfile("AfterImage.c"): #has not yet been compiled, so try to do so...
    cmd = "python setup.py build_ext --inplace"
    subprocess.call(cmd,shell=True)
#Import dependencies
import netStat as ns
import csv
import numpy as np
print("Importing Scapy Library")
from scapy.all import *
import os.path
import platform
import subprocess
import time
from scapy.layers.inet import *
from scapy.layers.inet6 import *
from scapy.layers.l2 import *
import numpy as np
import binascii
import random
import pandas as pd

#Extracts Kitsune features from given pcap file one packet at a time using "get_next_vector()"
# If wireshark is installed (tshark) it is used to parse (it's faster), otherwise, scapy is used (much slower).
# If wireshark is used then a tsv file (parsed version of the pcap) will be made -which you can use as your input next time
def get_list_from_csv(msg_path):
    msg = pd.read_csv(msg_path, keep_default_na=False)
    ab_msg = list()
    for i in range(len(msg)):
        ab_msg.append(dict(msg.iloc[i]))
    return ab_msg

def get_pcap_label(srcIP, srcproto, dstIP, dstproto, timestamp):
    ab_list = get_list_from_csv('./data/label_message.csv')
    for i in range(len(ab_list)):

        # STIME = ab_list[i]['srcTime']
        # DTIME = ab_list[i]['dstTime']
        SIP = ab_list[i]['srcIP']
        SPORT = ab_list[i]['srcPort']
        DIP = ab_list[i]['dstIP']
        DPORT = ab_list[i]['dstPort']
        # if timestamp < float(STIME) or timestamp > float(DTIME):
        #     continue
        if '' != SIP and srcIP != DIP:
            continue
        if '' != SPORT and srcproto != SPORT:
            continue
        if '' != DIP and dstIP != DIP:
            continue
        if '' != DPORT and dstproto != DPORT:
            continue
        print(srcIP, srcproto, dstIP, dstproto, timestamp)
        print(ab_list[i])
        return 1

    return 0

class FE:
    def __init__(self,file_path,limit=np.inf, data_type=None):
        self.path = file_path
        self.limit = limit
        self.parse_type = None #unknown
        self.curPacketIndx = 0
        self.tsvin = None #used for parsing TSV file
        self.scapyin = None #used for parsing pcap with scapy

        ### Prep pcap ##
        self.__prep__()

        ### Prep Feature extractor (AfterImage) ###
        maxHost = 100000000000
        maxSess = 100000000000
        self.nstat = ns.netStat(np.nan, maxHost, maxSess)
        self.data_type = data_type

    def _get_tshark_path(self):
        if platform.system() == 'Windows':
            return 'D:\Program Files (x86)\\Wireshark\\tshark.exe'
        else:
            system_path = os.environ['PATH']
            for path in system_path.split(os.pathsep):
                filename = os.path.join(path, 'tshark')
                if os.path.isfile(filename):
                    return filename
        return ''

    def __prep__(self):
        ### Find file: ###
        if not os.path.isfile(self.path):  # file does not exist
            print("File: " + self.path + " does not exist")
            raise Exception()

        ### check file type ###
        type = self.path.split('.')[-1]

        self._tshark = self._get_tshark_path()
        print(type)
        ##If file is TSV (pre-parsed by wireshark script)
        if self.parse_type == None:
            if type == "tsv":
                self.parse_type = "tsv"

            ##If file is pcap
            elif type == "pcap" or type == 'pcapng':
                # Try parsing via tshark dll of wireshark (faster)
                if os.path.isfile(self._tshark):
                    self.pcap2tsv_with_tshark()  # creates local tsv file
                    print("1")
                    self.path += ".tsv"
                    self.parse_type = "tsv"
                else: # Otherwise, parse with scapy (slower)
                    print("tshark not found. Trying scapy...")
                    self.parse_type = "scapy"
            else:
                print("File: " + self.path + " is not a tsv or pcap file")
                raise Exception()

        ### open readers ##
        if self.parse_type == "tsv":
            maxInt = sys.maxsize
            decrement = True
            while decrement:
                # decrease the maxInt value by factor 10
                # as long as the OverflowError occurs.
                decrement = False
                try:
                    csv.field_size_limit(maxInt)
                except OverflowError:
                    maxInt = int(maxInt / 10)
                    decrement = True

            print("counting lines in file...")
            num_lines = sum(1 for line in open(self.path))
            print("There are " + str(num_lines-1) + " Packets.")
            self.limit = min(self.limit, num_lines-1)
            self.tsvinf = open(self.path, 'rt', encoding="utf8")
            self.tsvin = csv.reader(self.tsvinf, delimiter='\t')
            row = self.tsvin.__next__() #move iterator past header

        else: # scapy
            print("Reading PCAP file via Scapy...")
            self.scapyin = rdpcap(self.path)
            self.limit = len(self.scapyin)
            print("Loaded " + str(len(self.scapyin)) + " Packets.")

    def get_next_vector(self):
        if self.curPacketIndx == self.limit:
            if self.parse_type == 'tsv':
                self.tsvinf.close()
            return [], []

        ### Parse next packet ###
        if self.parse_type == "tsv":
            row = self.tsvin.__next__()
            IPtype = np.nan
            timestamp = row[0]
            framelen = row[1]
            srcIP = ''
            dstIP = ''
            if row[4] != '':  # IPv4
                srcIP = row[4]
                dstIP = row[5]
                IPtype = 0
            elif row[17] != '':  # ipv6
                srcIP = row[17]
                dstIP = row[18]
                IPtype = 1
            srcproto = row[6] + row[8]  # UDP or TCP port: the concatenation of the two port strings will will results in an OR "[tcp|udp]"
            dstproto = row[7] + row[9]  # UDP or TCP port
            srcMAC = row[2]
            dstMAC = row[3]
            if srcproto == '':  # it's a L2/L1 level protocol
                if row[12] != '':  # is ARP
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = row[14]  # src IP (ARP)
                    dstIP = row[16]  # dst IP (ARP)
                    IPtype = 0
                elif row[10] != '':  # is ICMP
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                    srcIP = row[2]  # src MAC
                    dstIP = row[3]  # dst MAC

        elif self.parse_type == "scapy":
            packet = self.scapyin[self.curPacketIndx]
            IPtype = np.nan
            timestamp = packet.time
            print(timestamp)
            framelen = len(packet)
            if packet.haslayer(IP):  # IPv4
                srcIP = packet[IP].src
                dstIP = packet[IP].dst
                IPtype = 0
            elif packet.haslayer(IPv6):  # ipv6
                srcIP = packet[IPv6].src
                dstIP = packet[IPv6].dst
                IPtype = 1
            else:
                srcIP = ''
                dstIP = ''

            if packet.haslayer(TCP):
                srcproto = str(packet[TCP].sport)
                dstproto = str(packet[TCP].dport)
            elif packet.haslayer(UDP):
                srcproto = str(packet[UDP].sport)
                dstproto = str(packet[UDP].dport)
            else:
                srcproto = ''
                dstproto = ''

            srcMAC = packet.src
            dstMAC = packet.dst
            if srcproto == '':  # it's a L2/L1 level protocol
                if packet.haslayer(ARP):  # is ARP
                    srcproto = 'arp'
                    dstproto = 'arp'
                    srcIP = packet[ARP].psrc  # src IP (ARP)
                    dstIP = packet[ARP].pdst  # dst IP (ARP)
                    IPtype = 0
                elif packet.haslayer(ICMP):  # is ICMP
                    srcproto = 'icmp'
                    dstproto = 'icmp'
                    IPtype = 0
                elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                    srcIP = packet.src  # src MAC
                    dstIP = packet.dst  # dst MAC
        else:
            return [], []

        self.curPacketIndx = self.curPacketIndx + 1


        ### Extract Features
        try:
            return self.nstat.updateGetStats(IPtype, srcMAC, dstMAC, srcIP, srcproto, dstIP, dstproto, int(framelen), float(timestamp)), get_pcap_label( srcIP, srcproto, dstIP, dstproto, float(timestamp))
        except Exception as e:
            print(e)
            return [], []


    def pcap2tsv_with_tshark(self):
        print('Parsing with tshark...')
        fields = "-e frame.time_epoch -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport -e icmp.type -e icmp.code -e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 -e ipv6.src -e ipv6.dst"
        cmd =  '"' + self._tshark + '" -r '+ self.path +' -T fields '+ fields +' -E header=y -E occurrence=f > '+self.path+".tsv"
        print(cmd)
        if subprocess.call(cmd,shell=True):
            raise Exception("{} 执行失败".format(cmd))
        print("tshark parsing complete. File saved as: "+self.path +".tsv")

    def get_num_features(self):
        return len(self.nstat.getNetStatHeaders())

    def feature_extract_test(self):
        fvs = []
        cnt = 0
        label = []
        print("Start Feature Extracting. ")
        while(True):
            cnt += 1
            if cnt % 10000 == 0:
                print(cnt)
            x, y = self.get_next_vector()
            if len(x) == 0:
                break
            fvs.append(x)
            label.append(y)

        return np.array(fvs), np.array(label)

    def feature_extract_train(self):
        fvs_normal = []
        fvs_abnormal = []
        cnt = 0
        print("Start Feature Extracting. ")
        while(True):
            cnt += 1
            if cnt % 10000 == 0:
                print(cnt)
            x, y = self.get_next_vector()
            if len(x) == 0:
                break
            if 1 == y:
                fvs_abnormal.append(x)
            elif 0 == y:
                fvs_normal.append(x)
            else:
                print('label', y, 'is unknown')

        return np.array(fvs_normal), np.array(fvs_abnormal)

    def feature_extract(self):
        if self.data_type == 'train':
            return self.feature_extract_train()
        elif self.data_type == 'test':
            return self.feature_extract_test()
        else:
            print('data type', self.data_type, 'is unknown')

if __name__ == "__main__":
    # fe = FE("../data/IDS2017/merge22to23.pcap.tsv")
    fe = FE("./data/data.pcap", data_type='test')
    start = time.time()
    x, y = fe.feature_extract()
    with open('./data/data.pkl', 'wb') as f:  # write
        pickle.dump(x, f)
        f.close()
    with open('./data/data_label.pkl', 'wb') as f:  # write
        pickle.dump(y, f)
        f.close()

    stop = time.time()
    print("Time collapsed:", stop-start)
