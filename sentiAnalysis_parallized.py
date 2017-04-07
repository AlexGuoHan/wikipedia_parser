import pandas as pd
import multiprocessing
from sentiAnalysis import *


def load_apply_save(dataDir, outDir='outputFiles', chunksize=None, nworker=0):
	"""

	Load data, Apply cleaning / models, and Save
	For large data, we will read / clean chunk by chunk
	and use multiprocess to speed up

	"""

	# last_item are the last entry in the previous chunk
	cleaned_data = pd.DataFrame()
	assert nworker >= 0







class Workers:
	last_item = None
	DATA = None


	def __init__(self, dataIterator, logisticModel, mlpModel):
		self.dataIterator = dataIterator
		self.logisticModel = logisticModel
		self.mlpModel = mlpModel

	def update_data(self, newdata):
		''' update the last_item, and then update new chunk '''

		self.last_item = {
			'text': self.DATA.iloc[-1, :].text,
			'title': self.DATA.iloc[-1, :].title
		}
		
		self.DATA = newdata

	def process(self, nworkers):
		for 
		assert nworkers > 0, 'invalid nworker argument'
		pool = multiprocessing.Pool




	def process_by_title(self, title):
		data_subset = self.DATA[self.DATA.title == title]
		cleaned_data_subset = self.diff_clean_text(data_subset)
		cleaned_data_subset = self.apply_models_DF(cleaned_data,
													'logistic',
													self.logisticModel,
													['Added', 'Deleted'])
		cleaned_data_subset = self.apply_models_DF(cleaned_data,
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
								cleaned_text.split()
								))

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
		print('Starting ', multiprocessing.current_process().name)

	def apply_models_DF(self, model_name, model_dict, cleaned_text_cols):
		for col in cleaned_text_cols:
			texts = self.cleaned_data[col]
			for task, model in model_dict.items():
				scores = model.predict_proba(texts)[:,1]
				self.cleaned_data['%s_%s_%s_score'%(col, task, model_name)] = scores
