```bash
sudo docker run -it --rm \
    -v {path/to/dataset}:/dataset \
    -v {/path/to/abavit_gs_8.pth.tar}:/app/models/abavit_gs_8.pth.tar \
    abavitrack-cpu
```
edit just the left side of paths arguments starting with -v.
CAUTION: Never edit the right part of the paths arguments, which are the paths inside the container. For example, never edit /dataset or /app/models/abavit_gs_8.pth.tar.

then you will get a request like this
```
Aba‑ViTrack Finetuned & enhanced Submission Pipeline by Team: Zerone
Dataset root path [contest_release]: {/dataset}    
Manifest JSON path [/dataset/metadata/contestant_manifest.json]: {/dataset/metadata/contestant_manifest.json}
```
The answer should be relative to the '/dataset' root path, for example, if the manifest file is located at /dataset/metadata/contestant_manifest.json, you can just press enter to use the default path. If it is located at /dataset/metadata/my_manifest.json, you should input /dataset/metadata/my_manifest.json and then press enter.

after that you would be prompted to preprocess data, you have to input 'y' at least once, because we use our own folder structure.
```
preprocess data? [y/n]: y
```
> The preprocessed data will be stored in the same directory but will add the frames folder 'img' in each video folder, for example, if the original video is located at /dataset/video1.mp4, the preprocessed data will be stored at /dataset/video1/img.

```
Config name [abavit_gs_8]: 
Test epoch (checkpoint number) [17]:
Inference threads [8]: 
Results root directory [/app/outputs/tracking_results]: 
Output CSV file [/app/submission.csv]: 
```

