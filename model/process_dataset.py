import argparse
import multiprocessing
import time
import os
from process_video import VideoExtractor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


parser = argparse.ArgumentParser(description="Convert a dataset of videos into .npz file with keypoint data")
parser.add_argument("-n", "--num_processes", type=int, default=1)
parser.add_argument("-f", "--folder_path", type=str)
parser.add_argument("-o", "--output_path", type=str)

args = parser.parse_args()
num_processes = args.num_processes
folder_path = args.folder_path
output_path = args.output_path

os.makedirs(output_path, exist_ok=True)


def worker(queue):

    while True:
        try:
            filename = queue.get_nowait()
        except Exception as e:
            break
        
        file_path = os.path.join(folder_path, filename)
        
        if not os.path.isfile(file_path):  # Check if file exists
            continue
            
        name, extension = os.path.splitext(filename) 

        extractor = VideoExtractor(file_path)
        extractor.extract()
        extractor.save(os.path.join(output_path, name + '.npz'))


if __name__ == "__main__":

    processes = []
    queue = multiprocessing.Queue()
    total_files = 0

    for filename in os.listdir(folder_path):
        queue.put(filename)
        total_files += 1

    
    for i in range(num_processes):
        process = multiprocessing.Process(target=worker, args=(queue,))
        processes.append(process)
        process.start()

    while not queue.empty():
        print(f"Files processed: {total_files-queue.qsize()}/{total_files}", end="\r", flush=True)
        time.sleep(0.5)

    for process in processes:
        process.join()

    print(f"Finished processing {total_files} files.")

