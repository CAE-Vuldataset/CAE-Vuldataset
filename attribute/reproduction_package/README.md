# SV Data Quality Requirements

This codebase contains the data and scripts necessary for examining the data quality requirements of vulnerability datasets.

We have conducted our examination using 4 state of the art datasets from prior literature:  
| Dataset     | Link                                                            |
|-------------|-----------------------------------------------------------------|
| Big-Vul [1] | https://github.com/ZeoVan/MSR_20_Code_vulnerability_CSV_Dataset |
| Devign [2]  | https://sites.google.com/view/devign                            |
| D2A [3]     | https://github.com/ibm/D2A                                      |
| Juliet [4]  | https://samate.nist.gov/SARD/index.php                          |

## Setup

The datasets must first be downloaded. We have provided processed versions of the 4 above datasets that puts them in consistent formats. The datasets can be download from [HERE!](https://drive.google.com/file/d/1mVAPLd7VFasNpB-Dfhb8JbjIiwQAioqA/view?usp=sharing) Extract the datasets in the `dq_analysis/datasets/` folder.  

For impact analysis, we require a software vulnerability prediction model. This repository has been configured to use the [LineVul](https://github.com/awsm-research/LineVul) model [5]. Clone this repository and place the source code in the `dq_analysis/svp/LineVul/` folder.  

To identify code duplicates for the uniqueness attribute, we used the .NET version of code duplicate detector tool produced by Allamanis [6]. Setup:  
1-	Install .NET Core 2.1 or higher. [Link](https://dotnet.microsoft.com/en-us/download/dotnet/6.0).  
2-	Clone the [GitHub repository](https://github.com/Microsoft/near-duplicate-code-detector) to the main folder.  

## Installation
```
pip install -e .
```

## Usage
```
# Supported <attribute>
#  = 'accuracy', 'completeness', 'consistency', 'immediacy', 'uniqueness'
# Supported <dataset_name>
#  = 'Big-Vul', 'Devign', 'D2A', 'Juliet'

# Data quality measurement
./measure.sh <attribute> <dataset_name>

# Impact analysis
./impact.sh <attribute> <dataset_name>
```

Both data quality measurements and impact analysis require initial preparatory analysis, that is run before the measurements are taken. Immediacy must be run before uniqueness, as it requires the token preparation outputs.  

## Clean Datasets
We did not produce any _clean_ versions of the datasets, as best data cleaning practices require further consideration and formulation. However, duplicate and inconsistent data points can be easily removed with rule-based filters if necessary. Hence, we have provided these versions of the data under the `results/` folder of this repository. These datasets can be merged via the UID column.  


## References
[1] Fan, J., Li, Y., Wang, S. and Nguyen, T.N., 2020, June. A C/C++ Code Vulnerability Dataset with Code Changes and CVE Summaries. In Proceedings of the 17th International Conference on Mining Software Repositories (pp. 508-512).  
[2] Zhou, Y., Liu, S., Siow, J., Du, X. and Liu, Y., 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. Advances in neural information processing systems, 32.  
[3] Zheng, Y., Pujar, S., Lewis, B., Buratti, L., Epstein, E., Yang, B., Laredo, J., Morari, A. and Su, Z., 2021, May. D2a: A dataset built for ai-based vulnerability detection methods using differential analysis. In 2021 IEEE/ACM 43rd International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP) (pp. 111-120). IEEE.  
[4] Boland, T. and Black, P.E., 2012. Juliet 1. 1 C/C++ and java test suite. Computer, 45(10), pp.88-90.  
[5] Fu, M. and Tantithamthavorn, C., 2022. LineVul: A Transformer-based Line-Level Vulnerability Prediction. In 2022 IEEE/ACM 19th International Conference on Mining Software Repositories (MSR).  
[6] M. Allamanis, “The adverse effects of code duplication in machine learning models of code,” in Proceedings of the 2019 ACM SIGPLAN International Symposium on New Ideas, New Paradigms, and Reflections on Programming and Software, 2019, pp. 143–153.
