# SoK: Characterizing and Evaluating Software Vulnerability Datasets

## Description

Software vulnerabilities are the root cause of many cyber attacks. Not surprisingly, vulnerability-related studies, such as vulnerability analysis, detection, and repair, have received much attention. These studies have led to the introduction of many vulnerability datasets.  
However, there is no systematic understanding of the quality of these datasets.
This motivates the present SoK, which proposes a systematic characterization and evaluation of vulnerability datasets. Our systematization is based on four aspects of vulnerability datasets: their quality, purpose, source, and generation method; for measuring their quality, we present seven quantitative metrics, which may be of independent value.
We first conduct a study of 67 vulnerability datasets, including 57 academic publications and 10 industrial datasets, then we identify and systematize 28 vulnerability datasets that are publicly available.
Among other things, we find that there is no single vulnerability dataset or generation method of `` benchmark-quality``. We also discuss future research directions.

## Vulnerability Datasets

| Dataset | Paper | Venue/Organization | Link |
| --- | --- | --- | --- |
| VulDeePecker | VulDeePecker: A Deep Learning-Based System for Vulnerability Detection | NDSS | [链接](https://github.com/CGCL-codes/VulDeePecker) |
| SySeVR | SySeVR: A Framework for Using Deep Learning to Detect Software Vulnerabilities | TDSC | [链接](https://github.com/SySeVR/SySeVR) |
| VulDeeLocator | VulDeeLocator: A Deep Learning-based Fine-grained Vulnerability Detector | TDSC | [链接](https://github.com/VulDeeLocator/VulDeeLocator) |
| Reveal | Deep Learning based Vulnerability Detection: Are We There Yet? | TSE | [链接](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF) [链接2](https://github.com/VulDetProject/ReVeal)|
| Lin et al. | Deep learning-based vulnerable function detection: A benchmark | ICICS | [链接](https://github.com/Seahymn2019/Function-level-Vulnerability-Dataset/tree/master/Data) |
| Funded | Combining Graph-Based Learning With Automated Data Collection for Code Vulnerability Detection | TIFS | [链接](https://github.com/HuantWang/FUNDED_NISL) |
| Devign | Devign: Effective Vulnerability Identification by Learning Comprehensive Program Semantics via Graph Neural Networks | NeurIPS | [链接](https://sites.google.com/view/devign) |
| VulCNN | VulCNN: An Image-inspired Scalable Vulnerability Detection System | ICSE | [链接](https://github.com/CGCL-codes/VulCNN) |
| VUDDY | VUDDY: A Scalable Approach for Vulnerable Code Clone Discovery | S&P | [链接](https://github.com/squizz617/vuddy) |
| VulPecker | VulPecker: An Automated Vulnerability Detection System Based on Code Similarity Analysis | ACSAC | [链接](https://github.com/vulpecker/Vulpecker) |
| VDSimilar | VDSimilar: Vulnerability detection based on code similarity of vulnerabilities and patches | Computers & Security | [链接](https://github.com/sunhao123456789/siamese_dataset) |
| Magma | Magma: A Ground-Truth Fuzzing Benchmark | POMACS | [链接](https://github.com/HexHive/magma) |
| Lipp et al. | An Empirical Study on the Effectiveness of Static C Code Analyzers for Vulnerability Detection | ISSTA | [链接](https://doi.org/10.5281/zenodo.6515687) |
| SPI | SPI: Automated Identification of Security Patches via Commits | TOSEM | [链接](https://sites.google.com/view/du-commits/home) |
| Pan et al. | On the Use of Fine-grained Vulnerable Code Statements for Software Vulnerability Assessment Models | ICSE | [链接](https://figshare.com/articles/online_resource/TreeVul_-_Replication_Package/19727050) |
| Shi et al. | Does OpenBSD and Firefox’s Security Improve with Time? | TII | [链接](https://github.com/VulnSet/BetterOrWorse) |
| Liu et al. | A Large-Scale Empirical Study on Vulnerability Distribution within Projects and the Lessons Learned | ICSE | [链接](https://github.com/twelveand0/CarrotsBlender) |
| Detect0day | Detecting “0-Day” Vulnerability: An Empirical Study of Secret Security Patch in OSS | DSN | [链接](https://github.com/SecretPatch/Dataset) |
| CrossVul | CrossVul: a cross-language vulnerability dataset with commit data | ESEC/FSE | [链接](https://zenodo.org/record/4734050#.Y_YRtexBy3I) |
| PatchDB | PatchDB: A Large-Scale Security Patch Dataset | DSN | [链接](https://github.com/SunLab-GMU/PatchDB) |
| D2A | A Dataset Built for AI-Based Vulnerability Detection Methods Using Differential Analysis  | ICSE | [链接](https://github.com/IBM/D2A#using-the-dataset) |
| Ponta et al. | A Manually-Curated Dataset of Fixes to Vulnerabilities of Open-Source Software | MSR | [链接](https://github.com/SAP/project-kb/tree/main/MSR2019) |
| SECBENCH | SECBENCH: A Database of Real Security Vulnerabilities | SecSE@ESORICS | [链接](https://tqrg.github.io/secbench/ ) |
| Reis et al. | A ground-truth dataset of real security patches | NeurIPS | [链接](https://github.com/TQRG/security-patches-dataset) |
| BigVul | A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries | MSR | [链接](https://github.com/ZeoVan/MSR_20_Code_Vulnerability_CSV_Dataset) |
| CVEfixes | CVEfixes:Automated Collection of Vulnerabilities and Their Fixes from Open-Source Software  | PROMISE | [链接](https://github.com/secureIT-project/CVEfixes/blob/main/Doc/DataDictionary.md) |

## Folder Description
`./code/`: Script programs used for statistics are stored
