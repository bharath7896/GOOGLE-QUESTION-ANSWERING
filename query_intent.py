




nlp = spacy.load('en_core_web_md')

class query_intent(self):
	

	s = {'what','how','when','where','why','is','can','do','could','will','would','if','should','does'}
	

	def textract(phrase):
  		'''removing special charecters from the text'''
  		phrase = re.sub(r"won't", "will not", phrase)
  		phrase = re.sub(r"can\'t", "can not", phrase)
  		phrase = re.sub(r"\'s", " is", phrase)
 	 	phrase = re.sub(r"\'d", " would", phrase)
  		phrase = re.sub(r"\'ll", " will", phrase)
  		phrase = re.sub(r"\'t", " not", phrase)
  		phrase = re.sub(r"\'ve", " have", phrase)
  		phrase = re.sub(r"\'m", " am", phrase)
  		phrase = re.sub(r"/" , "or" , phrase)
  		phrase = re.sub('[^A-Za-z?]+', ' ', phrase)
  		return phrase

	def lemmatization(texts, allowed_postags=['PROPN','ADV','ADJ','VERB','NOUN']):
    		"""https://spacy.io/api/annotation"""
    		texts_out = []
    		nlp = spacy.load('en_core_web_sm')
    		for sent in texts:
        		doc = nlp(sent) 
        		texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags and token.text])
    		return texts_out

	
	def nq_q(nq):
    	        '''converting non question text to question format'''
    	        nq = pd.Series(nq)
    	        for txt in nq:
                	rst = txt.split()
                	doc = nlp(txt)
                  	for i,token in tqdm(enumerate(doc)):
                  		if token.text==rst[0] and token.pos_=='NOUN':  # adding what is/ what if to sentence starting with noun
                			nq = nq.replace(text,'what is '.join(text))
            			elif token.text==rst[0] and token.pos_ =='VERB':
                			nq = nq.replace(text,'how to '.join(text))
            			elif token.text==rst[0] and token.pos_ =='ADJ':
                			nq = nq.replace(text,'is '.join(text))
    		return nq  



	def qstart(qt):
		at = qt
    		(what,how,when,where,why,Is,can,do,will,If,should,does,are,nonq) = ([] for _ in range(14))
    		for i,qs in tqdm(enumerate(qt)):
        		qn = qs.lower().split() 
        		if qn[0].lower()=='what' and qn[-1][-1]=='?':
            			what.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'what')
        		elif qn[0].lower()=='how' and qn[-1][-1]=='?':
            			how.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'how')
        		elif qn[0].lower()=='when' and qn[-1][-1]=='?':
            			when.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'when')
        		elif qn[0].lower()=='where' and qn[-1][-1]=='?':
            			where.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'where')
        		elif qn[0].lower()=='why' and qn[-1][-1]=='?':
          			why.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'why')
        		elif qn[0].lower()=='is' and qn[-1][-1]=='?':
            			Is.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'is')
        		elif qn[0].lower()=='can' and qn[-1][-1]=='?':
           			can.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'can')
        		elif qn[0].lower()=='do' and qn[-1][-1]=='?':
            			do.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'do')
        		elif qn[0].lower()=='will' and qn[-1][-1]=='?':
            			will.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'will')
        		elif qn[0].lower()=='if' and qn[-1][-1]=='?':
            			If.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'if')
        		elif qn[0].lower()=='should' and qn[-1][-1]=='?':
            			should.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'should')
        		elif qn[0].lower()=='does' and qn[-1][-1]=='?':
            			does.append(qn[1])
            			at.values[i]=at.values[i].replace(at.values[i],'does')
			else:
            			nonq.append(qs)
            			at.values[i]=at.values[i].replace(at.values[i],'nonq')
    		dc = {'what':len(what)/len(qt),'when':len(when)/len(qt),'why':len(why)/len(qt),'where':len(where)/len(qt),
          	'how':len(how)/len(qt),'is':len(Is)/len(qt),'do':len(do)/len(qt),'can':len(can)/len(qt),'will':len(will)/len(qt),
          	'if':len(If)/len(qt),'should':len(should)/len(qt),'nonq':len(nonq)/len(qt)}
    		qt = qt.replace(dc)
    		return (dc,at)

	def pos_tag(sent):
		a = [];
    		txt = nlp(sent)
    		for token in txt:
        		a.append(token.pos_)
    		return a


	ps = {'VERB','PRON','DET','ADV'}

	def xscore(text):
    		res = []
    		doc = nlp(text.lower())
    		for i,token in tqdm(enumerate(doc)):
        		if token.pos_ in ps and token.text in s :
            			res.append(token.text)
            			return ' '.join(kw for kw in res)
        		else :
            			return text.lower()



	def mlp(num_hiddens, flatten):
    		net = nn.Sequential()
    		net.add(nn.Dropout(0.2))
    		net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    		net.add(nn.Dropout(0.2))
  		net.add(nn.Dense(num_hiddens, activation='relu', flatten=flatten))
    		return net

	def notq(qn):
    		qn = str(qn)
    		qsn = qn.lower().split()
    		if (qsn[-1][-1]!='?') or len(s.intersection(set(qsn)))==0:
        		return qn


	def cvr(qt):
    		nq = []
    		for cn in qt:
        		cn = cn.lower()
        		nq.append(notq(cn))
    		nql = [n for n in nq if n is not None ]
    		return nql


	def pr_key(key):
    		print('chc',stchc[key])
    		print('con',stcon[key])
    		print('def',stdef[key])
    		print('ins',stins[key])
    		print('pro',stpro[key])
    		print('ent',stent[key])
    		print('ops',stops[key])

	def tf_w2v(txt,tfidf_words):
    		tfidf_w2v_vector = [];
    		for sent in tqdm(txt):
        		vect = np.zeros(300)
        		tfidf_weight =0;
        		for word in sent.split():
            			if (word in glove_words) and (word in tfidf_words):
                			vec = model[word]
                			tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split()))#getting tfidf value for each word
                			vector += (vec * tf_idf) # calculating tfidf weight w2v
                			tf_idf_weight += tf_idf
    		if tfidf_weight !=0:
        		vector /= tfidf_weight
    		tfidf_w2v_vector.append(vector)
    		return tfidf_w2v_vector	


	@tf.function
	def plot_similarity(labels, features, rotation):
    		corr = scipy.stats.spearmanr(features, features)
    		sns.set(font_scale=1.2)
    		g = sns.heatmap(corr,
      			xticklabels=labels,
      			yticklabels=labels,
      			vmin=0,
      			vmax=1,
      			cmap="YlOrRd")
    		g.set_xticklabels(labels, rotation=rotation)
    		g.set_title("Semantic Textual Similarity")
	def run_and_plot(messages_):
    		message_embeddings_ = embed(messages_)
    		plot_similarity(messages_, message_embeddings_, 90)


	# mutual pos similarity
	def mpos_sim(pt):
    		mt=[]
    		for i in range(len(pt)):
        		for j in range(len(pt)):
            			if (i!=j):
                			while j==len(pt)-1:
                    				sm = edit_distance.SequenceMatcher(a=pt.values[i],b=pt.values[j])
                    				cnt = cnt
                				mt.append(sm.matches()/max([len(pt.values[i]),len(pt.values[j])]))
    		return mt


	#pos sequence
	def pos_seq(pt):
    		mt = []
    		pt = min([len(pt1),len(pt2)])
    		for i in range(pt):
        		sm = edit_distance.SequenceMatcher(a=pt1.values[i],b=pt2.values[i])
        		lp = sm.matches() / max([len(pt1.values[i]),len(pt2.values[i])])
        		mt.append(lp)
    		return mt


	@tf.function
	def spearman(y_true, y_pred):
     		return ( tf.py_function(scipy.stats.spearmanr, [tf.cast(y_pred, tf.float32),tf.cast(y_true, tf.float32)], Tout = tf.float32))


	def sim(a,b):
    		c = []
    		for i in range(len(a)):
        		c.append(np.inner(a[i],b[i]))
    		return np.array(c)


	def sent_score(col):
    		'''Calculating sentiment scores of the text data'''
    		neg,neu= [],[]
    		pos,comp = [],[]
   	 	sid = SentimentIntensityAnalyzer()
    		for txt in tqdm(col):
        		sps = sid.polarity_scores(txt)
        		neg.append(sps['neg'])
        		neu.append(sps['neu'])
        		pos.append(sps['pos'])
        		comp.append(sps['compound'])
        	dtp = {'neg':neg,'neu':neu,'pos':pos,'compound':comp}
    		return pd.DataFrame(dtp)


	def res_code(catcol,na,nb):
    		# catcol : categorical feature column name
    		# na : name of the positive class after response coding
    		# nb : name of the negative class after response coding
    		data = X_train[catcol]
    		a = dict(data.value_counts()) # total count of categorical variable
    		b = data.get_values() # total variables
   		c = pd.DataFrame({'cat':b,'y':Y_train}) # creating dataframe with cat variables and class response 
    		c_1 = dict(c.cat[c['y']==1].value_counts()) # tking positive class variables
   		k = [key for key in b if key not in c_1] # if certain key has count 1 in total and is not in positive class 
    		c_k = {x:0*i for i,x in enumerate(k) if len(k)!=0} # assigning the unfound category in the class with value 0
    		c_1.update(c_k) # updating the positive class response variable
    		d1 = np.array([float(c_1[key]/a[key]) for key in b ]) # finding the ressponse encode by findinf occurance ratio
    		d0 = 1 - d1; # occurance ratio for negative class as (p'=1-p)
    		t = []
    		k1 = {key : float(c_1[key]/a[key]) for key in b if key not in t}
    		k2 = {k:1-v for k,v in k1.items()}
    		dat = X_test[catcol]
    		b1 = dat.get_values()
    		a1 = {n:0.5 for n in b1 if n not in b}
    		k1.update(a1)
    		k2.update(a1)
    		t1 = dat.replace(k1)
    		t2 = dat.replace(k2)
    		return (pd.DataFrame({na:d1,nb:d0}),pd.DataFrame({na:t1,nb:t2}))















				
     






	