import os
import shutil


def main(dir, save):
    numbers = {}
    count=0
    subdirs = os.listdir(dir)
    for subdir in subdirs:
        subdir_path = os.path.join(dir, subdir)
        images = os.listdir(subdir_path)
        for image in images:
            number = image.split("_")[0]
            image_path = os.path.join(subdir_path, image)
            count += 1
            if number not in numbers.keys():
                numbers[number] = image_path
    numbers_key = list(numbers.keys())
    numbers_key.sort()
    print(len(numbers))
    print(count)
    index = 1

    num_dir = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014',
               '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029']
    print(len(num_dir))
    for pat in numbers_key:
        i = int(index/1000)
        save_dir = os.path.join(save, num_dir[i])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_path = os.path.join(save_dir, pat+'_'+str(index)+'.jpg')
        print(image_path)
        shutil.move(numbers[pat], image_path)
        index += 1


if __name__ == '__main__':

    list_dir = os.listdir('/media/yanjun/16D05224D0520A7F/numbertrain/mls_5000')

    for mlsname in list_dir:
        dir = '/media/yanjun/16D05224D0520A7F/numbertrain/mls_5000/'
        dir = dir + mlsname
        save = '/media/yanjun/16D05224D0520A7F/numbertrain/mls/'
        save = save + mlsname
        main(dir, save)