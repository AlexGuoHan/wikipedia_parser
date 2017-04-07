import pandas as pd
import multiprocessing
from sentiAnalysis import *


def load_apply_save(dataDir, outDir='outputFiles', chunksize=None, nworker=0, maxchunk=30000):
	"""

	Load data, Apply cleaning / models, and Save
	For large data, we will read / clean chunk by chunk
	and use multiprocess to speed up

	"""

	# last_item are the last entry in the previous chunk
	data_iterator = pd.read_csv(dataDir, sep='\t', chunksize=chunksize)
	filename = os.path.basename(dataDir)
	
	workers = Workers(data_iterator, 
						logisticModel,
						mlpModel,
						maxchunk,
						filename,
						outDir)
	workers.process(nworkers)



class Workers:
	def __init__(self, data_iterator,
				logisticModel, mlpModel,
				maxchunk, filename, outDir):
		""" Workers for Multiprocessing
		
		Args:
		    data_iterator : pandas.read_csv iterator
			DATA : the dataset chunk
		    last_item : the last row in the previous chunk
		    predicted_data : final output
		    data_titles : the unique title list in the chunk
		    maxchunk : max rows (chunks) to save and free memory
		    cumchunk : cumulative chunk size processed before freeing memory
		    outDir : directory to save file
		    filename : input file name, for saving files
		    fileidx : filename idx for output files

		"""
		self.data_iterator = data_iterator
		self.maxchunk = maxchunk
		self.filename = filename
		self.outDir = outDir
		
		self.logisticModel = logisticModel
		self.mlpModel = mlpModel
		
		self.DATA = None
		self.last_item = None
		self.predicted_DATA = pd.DataFrame()
		self.data_titles = None
		self.cumchunk = 0
		self.fileidx = 0



	def update_data(self, newdata):
		""" Update data chunk after each iteration """
		self.DATA = newdata
		self.data_titles = newdata.title.unique()
		self.cumchunk += self.data_iterator.chunksize

	def update_last_item(self):
		self.last_item = {
			'text': self.DATA.iloc[-1, :].text,
			'title': self.DATA.iloc[-1, :].title
		}

	def process(self, nworkers):
		"""Multiprocess over workers, store the result in self.predicted_Data
		
		Args:
		    nworkers : number of workes to parallelize on
		
		"""
		assert nworkers > 0, 'invalid nworker argument'
		assert nworkers <= multiprocessing.cpu_count(), 'too many workers'

		# read data chunk by chunk and process by title
		for data in self.data_iterator:
			tic.go('Start Iteration')
			# update the data stored in the field for other functions to access
			self.update_data(data)
			# create a pool of workers, initialize with a msg
			pool = multiprocessing.Pool(processes=nworker,
										initializer=worker_initializer)
			# parallelize the job to pool of workers
			predicted_data_subsets = pool.map(process_by_title, self.dataTitles)
			pool.close()
			pool.join()
			# append the results from pool to the data
			self.predicted_DATA = self.predicted_DATA.append(predicted_data_subsets)
			tic.stop()
			# save if exceeds maxchunk
			self.save_data()
			# update last_item
			self.update_last_item()

		self.save_data(end_of_file=True)




	def process_by_title(self, title):
		"""Process the files by title
		
		Args:
		    title : iterable list
		
		Returns:
		    dataframe with cleaned text and predicted scores 
		"""
		data_subset = self.DATA[self.DATA.title == title]
		cleaned_data_subset = self.diff_clean_text(data_subset)

		assert 'Added' in cleaned_data_subset.columns, 'Data Format Error, no Added'
		assert 'Deleted' in cleaned_data_subset.columns, 'Data Format Error, no Deleted'

		cleaned_data_subset = self.apply_models_DF(cleaned_data_subset,
													'logistic',
													self.logisticModel,
													['Added', 'Deleted'])
		cleaned_data_subset = self.apply_models_DF(cleaned_data_subset,
													'mlp',
													self.mlpModel,
													['Added', 'Deleted'])
		return cleaned_data_subset




	def diff_clean_text(self, data):
		''' Taking the difference between two history version '''
		
		# load the last item
		try:
			title_prev = self.last_item['title']
			cleaned_text_prev = self.last_item['text']
		except TypeError:
			# the last item is None
			# this means there is no last item to compare with
			# simply ignore it
			pass


		Differ = difflib.Differ()
		ADDED = []
		ADDED_bytes = []
		DELETED = []
		DELETED_bytes = []
		
		for row in data.iterrows():
			# row = [ [idx] [content] ]
			content = row[1]
			assert 'title' in content.keys(), 'Data Format Error, no Title'
			assert 'text' in content.keys(), 'Data Format Error, no Text'

			title = content['title']
			cleaned_text = clean_text(content['text'])


			try:
				if(title == title_prev):
					diff = list(Differ.compare(
						cleaned_text_prev.split(),
						cleaned_text.split()   ))

					add_words = []
					del_words = []
					for d in diff:
						if(d[0] == '+'):
							add_words.append(d[2:])
						elif(d[0] == '-'):
							del_words.append(d[2:])
					added = ' '.join(add_words)
					deleted = ' '.join(del_words)

				else:
					# at the beginning of new page
					added = cleaned_text
					deleted = ''

			except NameError:
				# at the beginning of the file
				# no title_prev or cleaned_text_prev
				added = cleaned_text
				deleted = ''

			ADDED.append(added)
			ADDED_bytes.append(len(added))
			DELETED.append(deleted)
			DELETED_bytes.append(len(deleted))

			title_prev = title
			cleaned_text_prev = cleaned_text

		data['Added'] = ADDED
		data['Added_Bytes'] = ADDED_bytes
		data['Deleted'] = DELETED
		data['Deleted_Bytes'] = DELETED_bytes

		return data


	def worker_initializer():
		"""Initialize when creating a pool of workers"""
		print('Starting ', multiprocessing.current_process().name)

	def apply_models_DF(self, model_name, model_dict, cleaned_text_cols):
		"""Apply sentiment analysis models on dataframe
		
		Args:
		    model_name : logistic or mlp
		    model_dict : model dictionary, 
		    			self.logisticModel or self.mlpModel
		    cleaned_text_cols : the cleaned columns to make predictions on
		
		Returns:
		    dataframe with predicted scores 
		"""
		for col in cleaned_text_cols:
			texts = self.cleaned_data[col]
			for task, model in model_dict.items():
				scores = model.predict_proba(texts)[:,1]
				self.cleaned_data['%s_%s_%s_score'%(col, task, model_name)] = scores


	def save_data(self, end_of_file=False):
		"""Save data when 1) exceeds maxchunk 2) end of file"""
		if(self.cumchunk >= self.maxchunk or end_of_file is True):
			# if cumulative chunk size exceeds max chunk defined by user
			# of if at the end of file
			print('NOW SAVING FILES')
			# make directory if output folder does not exist
			if(os.path.isdir(self.outDir) is False):
				os.makedirs(self.outDir)
			# save the file
			this.predicted_DATA.to_csv(os.path.join(self.outDir,
				'predicted_%s_%d' % (self.filename, self.fileidx)), sep='\t')
			# empty the predicted data to free memory
			this.predicted_DATA = pd.DataFrame()
		else:
			pass
