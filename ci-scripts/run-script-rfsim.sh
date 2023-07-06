#!/bin/bash

ranBranch=develop
ranCommitID=c81ed57a9303f1c0b842051956c730beced9997e

eNBIPAddress=none
eNBUserName=oaicicd
eNBPassword=DOESNOTMATTER
eNB1IPAddress=none
eNB1UserName=oaicicd
eNB1Password=DOESNOTMATTER
eNB2IPAddress=none
eNB2UserName=oaicicd
eNB2Password=DOESNOTMATTER
UEIPAddress=none
UEUserName=oaicicd
UEPassword=DOESNOTMATTER
EPCIPAddress=192.168.107.17
EPCType=ltebox
EPCUserName=oaicicd
EPCPassword=DOESNOTMATTER

python3 main.py \
  --mode=InitiateHtml \
  --ranRepository=https://gitlab.eurecom.fr/oai/openairinterface5g.git \
  --ranBranch=$ranBranch --ranCommitID=$ranCommitID --ranTargetBranch=$ranTargetBranch \
  --ranAllowMerge=true \
  --XMLTestFile=xml_files/container_5g_rfsim.xml \
  --XMLTestFile=xml_files/container_5g_rfsim_down.xml \
  --XMLTestFile=xml_files/container_5g_fdd_rfsim.xml \
  --XMLTestFile=xml_files/container_5g_fdd_rfsim_down.xml \
  --XMLTestFile=xml_files/container_5g_f1_rfsim.xml \
  --XMLTestFile=xml_files/container_5g_f1_rfsim_down.xml \
  --XMLTestFile=xml_files/container_5g_e1_rfsim.xml \
  --XMLTestFile=xml_files/container_5g_e1_rfsim_down.xml \
  --XMLTestFile=xml_files/container_5g_rfsim_24prb.xml \
  --XMLTestFile=xml_files/container_5g_rfsim_24prb_down.xml \
  --XMLTestFile=xml_files/container_5g_rfsim_fr2_32prb.xml \
  --XMLTestFile=xml_files/container_5g_rfsim_fr2_32prb_down.xml \
  --XMLTestFile=xml_files/container_5g_rfsim_fdd_phytest.xml \
  --XMLTestFile=xml_files/container_5g_rfsim_fdd_phytest_down.xml \
  --XMLTestFile=xml_files/container_5g_rfsim_tdd_dora.xml \
  --XMLTestFile=xml_files/container_5g_rfsim_tdd_dora_down.xml \
  --XMLTestFile=xml_files/container_5g_rfsim_u0_25prb.xml \
  --XMLTestFile=xml_files/container_5g_rfsim_u0_25prb_down.xml

python3 main.py \
  --mode=TesteNB \
  --ranRepository=https://gitlab.eurecom.fr/oai/openairinterface5g.git \
  --ranBranch=$ranBranch --ranCommitID=$ranCommitID --ranTargetBranch=$ranTargetBranch \
  --ranAllowMerge=true \
  --eNBIPAddress=$eNBIPAddress --eNBUserName=$eNBUserName --eNBPassword=$eNBPassword \
  --eNBSourceCodePath=/tmp/CI_phy_sim_test \
  --EPCIPAddress=$eNBIPAddress --EPCType=OAI-Rel14-Docker --EPCUserName=$eNBUserName --EPCPassword=$eNBPassword \
  --EPCSourceCodePath=/tmp/CI-epc-test \
  --XMLTestFile=xml_files/container_5g_rfsim_2x2.xml
 # --XMLTestFile=xml_files/container_5g_e1_rfsim.xml
  #--XMLTestFile=xml_files/container_5g_rfsim_24prb.xml
  #--XMLTestFile=xml_files/container_5g_rfsim_fr2_32prb.xml
  #--XMLTestFile=xml_files/container_5g_rfsim_2x2.xml
  #--XMLTestFile=xml_files/container_5g_rfsim_fdd_phytest.xml
  #--XMLTestFile=xml_files/container_5g_rfsim_u0_25prb.xml
  #--XMLTestFile=xml_files/container_5g_fdd_rfsim.xml

#python3 main.py \
#  --mode=TesteNB \
#  --ranRepository=https://gitlab.eurecom.fr/oai/openairinterface5g.git \
#  --ranBranch=$ranBranch --ranCommitID=$ranCommitID --ranTargetBranch=$ranTargetBranch \
#  --ranAllowMerge=true \
#  --eNBIPAddress=$eNBIPAddress --eNBUserName=$eNBUserName --eNBPassword=$eNBPassword \
#  --eNBSourceCodePath=/tmp/CI_phy_sim_test \
#  --EPCIPAddress=$eNBIPAddress --EPCType=OAI-Rel14-Docker --EPCUserName=$eNBUserName --EPCPassword=$eNBPassword \
#  --EPCSourceCodePath=/tmp/CI-epc-test \
#  --XMLTestFile=xml_files/container_5g_rfsim_tdd_dora_down.xml

python3 main.py \
  --mode=FinalizeHtml \
  --finalStatus=false \
  --eNBIPAddress=$eNBIPAddress --eNBUserName=$eNBUserName --eNBPassword=$eNBPassword
