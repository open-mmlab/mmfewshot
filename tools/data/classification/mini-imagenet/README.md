# Preparing Mini-ImageNet Dataset

<!-- [DATASET] -->

```bibtex
@inproceedings{ren18fewshotssl,
    author = {Mengye Ren and Eleni Triantafillou and Sachin Ravi and Jake Snell and Kevin Swersky and Joshua B. Tenenbaum and Hugo Larochelle and Richard S. Zemel},
    title = {Meta-Learning for Semi-Supervised Few-Shot Classification},
    booktitle = {Proceedings of 6th International Conference on Learning Representations {ICLR}},
    year = {2018},
}

@article{ILSVRC15,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = {{ImageNet Large Scale Visual Recognition Challenge}},
    Year = {2015},
    journal = {International Journal of Computer Vision (IJCV)},
    doi = {10.1007/s11263-015-0816-y},
    volume = {115},
    number = {3},
    pages = {211-252}
}
```

The split of mini-imagenet can be found from [here](https://github.com/twitter-research/meta-learning-lstm/tree/master/data/miniImagenet).
The whole imagenet dataset can be downloaded from [here](https://drive.google.com/open?id=1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07).

The data structure is as follows:
```text
mmfewshot
├── mmfewshot
├── configs
├── data
│   ├── mini_imagenet
│   │   ├── images
│   │   │   ├── n01440764
│   │   │   ├── n01443537
...
```
