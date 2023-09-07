# EPFNet
Enhanced Point Feature Network for Point Cloud Salient Object Detection
# Abstract
Point cloud salient object detection (SOD) aims to identify and segment the most prominent areas or targets in a 3D scene. Currently, research on point cloud SOD is still in its infancy, with most approaches neglecting the color information available in the point cloud. In this paper, we propose an enhanced point feature network (EPFNet) for point cloud SOD. Firstly, we extract RGB information from the point cloud with color and use it as input for the dual-stream network. Next, we introduce a multi-scale information enhancement (MIE) module to enhance common information and embed complementary information, acquiring RGB features at different scales and transforming them into a point feature. To gain access to global semantic information, we propose a point-image fusion (PIF)
module, which aggregates the enhanced point features with the feature-rich RGB features and produces the final results. We conduct extensive experiments to validate the effectiveness of EPFNet and our approach outperforms 7 other state-of-the-art models on 4 metrics.
<img width="2274" alt="overview" src="https://github.com/ELOESZHANG/EPFNet/assets/46095890/bff79a74-43ff-4588-b031-54058eba8ebf">
# Qualitative Results

<img width="1000" alt="Qualitative Comparison（压缩）" src="https://github.com/ELOESZHANG/EPFNet/assets/46095890/cba57998-cbe7-4aea-af50-2bb181306a49">



# Reference
Part of the code is borrowed from the work of the following authors:

1.https://git.openi.org.cn/OpenPointCloud/PCSOD

2.https://github.com/xuebinqin/U-2-Net
