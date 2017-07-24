import h5py
from os.path import join

if __name__ == '__main__':
	
	dir_list = ['french_gmm','mandarin_gmm']

	for dir_name in dir_list:
		file_list = ['120s.h5f']
		
		try:	
			for h5file in file_list:
				h5 = h5py.File(join(dir_name, h5file), 'r+')
				#h5['/features/MFCC'] = h5['/features/features']
				del(h5['/features/features'])

		except:
			continue



