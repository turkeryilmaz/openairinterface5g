#!/bin/bash

ranBranch=ci-iperf-improvements
ranCommitID=8f9bf35178a2b09b7ab268ea8d274ee30f9e447e

eNBIPAddress=172.21.16.109
eNBUserName=oaicicd
eNBPassword=DOESNOTMATTER
eNB1IPAddress=172.21.16.109
eNB1UserName=oaicicd
eNB1Password=DOESNOTMATTER
eNB2IPAddress=172.21.16.109
eNB2UserName=oaicicd
eNB2Password=DOESNOTMATTER
UEIPAddress=172.21.16.137
UEUserName=oaicicd
UEPassword=DOESNOTMATTER
EPCIPAddress=172.21.16.137
EPCType=ltebox
EPCUserName=oaicicd
EPCPassword=DOESNOTMATTER

python3 main.py \
  --mode=InitiateHtml \
  --ranRepository=https://gitlab.eurecom.fr/oai/openairinterface5g.git \
  --ranBranch=$ranBranch --ranCommitID=$ranCommitID --ranTargetBranch=$ranTargetBranch \
  --ranAllowMerge=true \
  --XMLTestFile=xml_files/fr1_5gc_closure.xml \
  --XMLTestFile=xml_files/fr1_5gc_start.xml \
  --XMLTestFile=xml_files/container_sa_b200_quectel.xml \
  --XMLTestFile=xml_files/container_sa_b200_terminate.xml \
  --XMLTestFile=xml_files/container_sa_f1_b200_quectel.xml \
  --XMLTestFile=xml_files/container_sa_f1_b200_terminate.xml \
  --XMLTestFile=xml_files/fr1_5gc_closure.xml \

python3 main.py \
  --mode=TesteNB \
  --ranRepository=https://gitlab.eurecom.fr/oai/openairinterface5g.git \
  --ranBranch=$ranBranch --ranCommitID=$ranCommitID --ranTargetBranch=$ranTargetBranch \
  --ranAllowMerge=true \
  --eNBIPAddress=$eNBIPAddress --eNBUserName=$eNBUserName --eNBPassword=$eNBPassword \
  --eNBSourceCodePath=/tmp/CI_phy_sim_test \
  --EPCIPAddress=$eNBIPAddress --EPCType=OAI-Rel14-Docker --EPCUserName=$eNBUserName --EPCPassword=$eNBPassword \
  --EPCSourceCodePath=/tmp/CI-epc-test \
  --XMLTestFile=xml_files/container_sa_b200_quectel.xml

python3 main.py \
  --mode=FinalizeHtml \
  --finalStatus=false \
  --eNBIPAddress=$eNBIPAddress --eNBUserName=$eNBUserName --eNBPassword=$eNBPassword
