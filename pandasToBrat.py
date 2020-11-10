import re
import os
import pandas as pd
import numpy as np
from pandasToBrat.extract_tools import default_tokenizer

def _getDictionnaryKeys(dictionnary):
    """
        Function that get keys from a dict object and flatten sub dict.
    """
    
    keys_array = []
    for key in dictionnary.keys():
        keys_array.append(key)
        if (type(dictionnary[key]) == type({})):
            keys_array = keys_array+_getDictionnaryKeys(dictionnary[key])
    return(keys_array)

class pandasToBrat:

    """
        Class for Pandas brat folder management.
        For each brat folder, there is an instance of pandasToBrat.
        It supports importation and exportation of configurations for relations and entities.
        Documents importation and exportation.
        Annotations and entities importation and exportation.

        Inputs :
            folder, str : path of brat folder
    """
    
    def __init__(self, folder):
        self.folder = folder
        self.conf_file = 'annotation.conf'
        
        self.emptyDFCols = {
            "annotations":["id","type_id", "word", "label", "start", "end"],
            "relations":["id","type_id","relation","Arg1","Arg2"]
        }
        
        # Adding '/' to folder path if missing
        if(self.folder[-1] != '/'):
            self.folder += '/'
        
        # Creating folder if do not exist
        if (os.path.isdir(self.folder)) == False:
            os.mkdir(self.folder)
            
        # Loading conf file if exists | creating empty conf file if not
        self.read_conf()
            
    def _emptyData(self):
        fileList = self._getFileList()
        nb_files = fileList.shape[0]
        
        confirmation = input("Deleting all data ({} files), press y to confirm :".format(nb_files))
        if confirmation == 'y':
            fileList["filename"].apply(lambda x: os.remove(self.folder+x))
            print("{} files deleted.".format(nb_files))
            
    def _generateEntitiesStr (self, conf, data = '', level = 0):
        
        if (type(conf) != type({})):
            return data
        
        # Parsing keys
        for key in conf.keys():
            value = conf[key]

            if value == True:
                data += '\n'+level*'\t'+key
            elif value == False:
                data += '\n'+level*'\t'+'!'+key
            elif type(value) == type({}):
                data += '\n'+level*'\t'+key
                data = self._generateEntitiesStr(value, data, level+1)

        return data
    
    def _writeEntitiesLevel (self, conf, data, last_n = -1):
        
        for n in range(last_n,len(conf)):
            # If empty : pass, if not the last line : pass
            if (conf[n] != '' and n > last_n):
                level = len(conf[n].split("\t"))-1
                if (n+1 <= len(conf)): # Level of next item
                    next_level = len(conf[n+1].split("\t"))-1
                else:
                    next_level = level
                    
                splitted_str = conf[n].split("\t")
                str_clean = splitted_str[len(splitted_str)-1]
                
                if (level >= next_level): # On écrit les lignes de même niveau
                    if (str_clean[0] == '!'):
                        data[str_clean[1:]] = False
                    else:
                        data[str_clean] = True
                    
                    if (level > next_level):
                        # On casse la boucle
                        break
                elif (level < next_level): # On écrit les lignes inférieurs par récurence
                    splitted_str = conf[n].split("\t")
                    last_n, data[str_clean] = self._writeEntitiesLevel(conf, {}, n)

        return(n, data)
    
    def _readRelations(self, relations, entities = []):
        data = {}

        for relation in relations.split("\n"):
            if relation != '':
                relation_data = relation.split("\t")[0]
                args = list(map(lambda x: x.split(":")[1], relation.split("\t")[1].split(", ")))
                args_valid = list(filter(lambda x: x in entities, args))

                if (len(args_valid) > 0):
                    data[relation_data] = {"args":args_valid}
                    
        return data
    
    def _writeRelations(self, relations, entities = []):
        data = ''
        for relation in relations:
            args_array = list(filter(lambda x: x in entities, relations[relation]["args"]))

            if (len(args_array) > 0):
                data += '\n'+relation+'\t'

                for n in range(0, len(args_array)):
                    data += int(bool(n))*', '+'Arg'+str(n+1)+':'+args_array[n]
                    
        return data
    
    def read_conf (self):
        """
            Get the current Brat configuration.
            Output :
                Dict containing "entities" and "relations" configurations.
        """
        
        if (os.path.isfile(self.folder+self.conf_file)):
            
            # Reading file
            file = open(self.folder+self.conf_file)
            conf_str = file.read()
            file.close()
            
            # Splitting conf_str
            conf_data = re.split(re.compile(r"\[[a-zA-Z]+\]", re.DOTALL), conf_str)[1:]
            
            data = {}
            
            # Reading enteties
            data["entities"] = self._writeEntitiesLevel(conf_data[0].split("\n"), {})[1]
            
            # Reading relations
            entitiesKeys = _getDictionnaryKeys(data["entities"])
            data["relations"] = self._readRelations(conf_data[1], entitiesKeys)
            
            return(data)
            
        else:
            self.write_conf()
            self.read_conf()
    
    def write_conf(self, entities = {}, relations = {}, events = {}, attributes = {}):
        """
            Write or overwrite configuration file.
            It actually doesn't suppport events and attributes configuration data.

            inputs :
                entities, dict : dict containing the entities. If an entities do have children, his value is an other dict, otherwise, it is set as True.
                relations, dict : dict containing the relations between entities, each key is a relation name, the value is a dict with a "args" key containing the list of related entities.
        """
        
        # TODO : Add events and attributes support.

        conf_str = ''
        
        # Entities
        conf_str += '\n\n[entities]'
        conf_str += self._generateEntitiesStr(entities)
        
        # relations
        conf_str += '\n\n[relations]'
        entitiesKeys = _getDictionnaryKeys(entities)
        conf_str += self._writeRelations(relations, entitiesKeys)
        
        # attributes
        conf_str += '\n\n[attributes]'

        # events
        conf_str += '\n\n[events]'
        
        # Write conf file
        file = open(self.folder+self.conf_file,'w')
        file.write(conf_str)
        file.close()
        
    def _getFileList(self):
        # Listing files
        filesDF = pd.DataFrame({'filename':pd.Series(os.listdir(self.folder))})
        filesDFSplitted = filesDF["filename"].str.split(".", expand = True)
        filesDF["id"] = filesDFSplitted[0]
        filesDF["filetype"] = filesDFSplitted[1]
        
        filesDF = filesDF[filesDF["filetype"].isin(["txt","ann"])]
        
        return(filesDF)
        
    def _parseData(self):
        
        # Listing files
        filesDF = self._getFileList()
        
         # Getting data from txt and ann
        filesDF_txt = filesDF.rename(columns = {"filename":"text_data"}).loc[filesDF["filetype"] == "txt", ["id","text_data"]]
        filesDF_ann = filesDF.rename(columns = {"filename":"annotation"}).loc[filesDF["filetype"] == "ann", ["id","annotation"]]
        dataDF = filesDF_txt.join(filesDF_ann.set_index("id"), on = "id")
        dataDF["text_data"] = dataDF["text_data"].apply(lambda x: open(self.folder+x).read())
        dataDF["annotation"] = dataDF["annotation"].apply(lambda x: open(self.folder+x).read())
        
        return(dataDF)

    def read_text(self):

        """
            read_text
            Get a pandas DataFrame containing the brat documents.

            Input : None
            Output : Pandas dataframe
        """
        
        dataDF = self._parseData()
                
        return(dataDF[["id","text_data"]])

    def read_annotation(self, ids = []):

        """
            read_annotation
            Get annotations from the brat folder.
            You can get specific annotation by filtering by id.

            input :
                ids, list (optionnal) : list of id for which you want the annotation data, if empty all annotations are returned.

            output :
                dict containing an annotations and relations data.
        """
        
        data = {}
        data["annotations"] = pd.DataFrame(columns=self.emptyDFCols["annotations"])
        data["relations"] = pd.DataFrame(columns=self.emptyDFCols["relations"])
        
        dataDF = self._parseData()[["id","annotation"]]
        dataDF = dataDF[(dataDF["annotation"].isna() == False) & (dataDF["annotation"] != '')] # Removing empty annotation
        
        # Filtering by ids
        if (len(ids) > 0):
            dataDF = dataDF[dataDF["id"].isin(pd.Series(ids).astype(str))]
            
        if (dataDF.shape[0] > 0):
            
            # Ann data to pandas
            dataDF = dataDF.join(dataDF["annotation"].str.split("\n").apply(pd.Series).stack().reset_index(level = 0).set_index("level_0")).reset_index(drop = True).drop("annotation", axis = 1).rename(columns = {0: "annotation"})
            dataDF = dataDF[dataDF["annotation"].str.len() > 0].reset_index(drop = True)
            dataDF = dataDF.join(dataDF["annotation"].str.split("\t", expand = True).rename(columns = {0: 'type_id', 1: 'data', 2: 'word'})).drop("annotation", axis = 1)
            dataDF["type"] = dataDF["type_id"].str.slice(0,1)
            
            ## Annotations
            data["annotations"] = dataDF[dataDF["type"] == 'T']
            if (data["annotations"].shape[0] > 0):
                data["annotations"] = data["annotations"].join(data["annotations"]["data"].str.split(" ", expand = True).rename(columns = {0: "label", 1: "start", 2: "end"})).drop(columns = ["data","type"])

            ## Relations
            data["relations"] = dataDF[dataDF["type"] == 'R']

            if (data["relations"].shape[0] > 0):
                tmp_splitted = data["relations"]["data"].str.split(" ", expand = True).rename(columns = {0: "relation"})

                ### Col names
                rename_dict = dict(zip(list(tmp_splitted.columns.values[1:]), list("Arg"+tmp_splitted.columns.values[1:].astype(str).astype(object))))
                tmp_splitted = tmp_splitted.rename(columns = rename_dict)

                ### Merging data
                tmp_splitted = tmp_splitted[["relation"]].join(tmp_splitted.loc[:,tmp_splitted.columns[tmp_splitted.columns != 'relation']].applymap(lambda x: x.split(":")[1]))
                data["relations"] = data["relations"].join(tmp_splitted).drop(columns = ["data","type","word"])
        
        return(data)
    
    def _write_function(self, x, filetype = "txt", overwrite = False):
        
        filenames = []
        
        if (filetype == 'txt' or filetype == 'both'):
            filenames.append(self.folder+str(x["filename"])+'.txt')
            
        if (filetype == 'ann' or filetype == 'both'):
            filenames.append(self.folder+str(x["filename"])+'.ann')
        
        for filename in filenames:
            try:
                open(str(filename), "r")
                is_file = True
            except FileNotFoundError:
                is_file = False
                        
            if ((is_file == False) or (overwrite == True)):
                file = open(str(filename), "w")
                file.write(x["content"])
                file.close()
    
    def write_text(self, text_id, text, empty = False, overWriteAnnotations = False):
        
        """
            write_text
            Send text data from the brat folder.

            input :
                text_id, pd.Series : pandas series containing documents ids
                text, pd.Series : pandas series containing documents text in the same order as text_id
                empty, boolean : if True the brat folder is emptyied of all but configuration data (text and ann files) before writting
                overwriteAnnotations, boolean : if True, the current annotation files are replaced by blank one
        """

        if overWriteAnnotations == True: # On controle la façon dont la variable est écrite
            overwriteAnn = True
        else:
            overwriteAnn = False
        
        if (type(text) == type(pd.Series()) and type(text_id) == type(pd.Series()) and text.shape[0] == text_id.shape[0]):
            
            # ID check : check should be smaller than text : check if not inverted
            if (text_id.astype(str).str.len().max() < text.astype(str).str.len().max()):
                
                # empty : option to erase existing data
                if (empty):
                    self._emptyData()

                # Writting data
                print("Writting data")
                df_text = pd.DataFrame({"filename":text_id, "content":text})
                df_ann = pd.DataFrame({"filename":text_id, "content":""})
                
                df_text.apply(lambda x: self._write_function(x, filetype = "txt", overwrite = True), axis = 1)
                df_ann.apply(lambda x: self._write_function(x, filetype = "ann", overwrite = overwriteAnn), axis = 1)
                print("data written.")
                
            else:
                raise ValueError('ID is larger than text, maybe you inverted them.')
        
        else:
            raise ValueError('Incorrect variable type, expected two Pandas Series of same shape.')
                
    def write_annotations(self, df, text_id, word, label, start, end, overwrite = False):
        
        """
            write_annotations
            Send annotation data from the brat folder. Useful to pre-anotate some data.

            input :
                df, pd.Dataframe : dataframe containing annotations data, should contains the text id, the annotated word, the annotated label, the start and end offset.
                text_id, str : name of the column in df which contains the document id
                word, str : name of the column in df which contains the annotated word
                label, str : name of the column in df which contains the label of the annotated word
                start, str : name of the column in df which contains the start offset
                end, str : name of the column in df which contains the end offset
                overwrite, boolean : if True, the current annotation files are replaced by new data, otherwise, the new annotations are merged with existing one
        """
        
        # Checking data types
        if (type(df) == type(pd.DataFrame())):
            
            # Loading df
            df = df.rename(columns = {text_id:"id",word:"word",label:"label",start:"start",end:"end"})
            df["type_id"] = df.groupby("id").cumcount()+1
            
            # List of ids
            ids = df["id"].unique()

            # Loading current data
            current_annotation = self.read_annotation(ids)            
            current_annotations = current_annotation["annotations"]
            tmaxDFAnnotations = current_annotations.set_index(["id"])["type_id"].str.slice(1,).astype(int).reset_index().groupby("id").max().rename(columns = {"type_id":"Tmax"})
            
            if (overwrite == True):
                df["type_id"] = "T"+df["type_id"].astype(str)
                new_annotations = df
            else:
                df = df.join(tmaxDFAnnotations, on = "id").fillna(0)
                df["type_id"] = "T"+(df["type_id"]+df["Tmax"]).astype(int).astype(str)
                df = df.drop(columns = ["Tmax"])
                
                new_annotations = pd.concat((current_annotations, df[self.emptyDFCols["annotations"]])).reset_index(drop = True)
            
            new_annotations.drop_duplicates() ## Removing duplicates
            
            # Injecting new annotations
            current_annotation["annotations"] = new_annotations
            
            # Calling write function
            self._write_annotation(current_annotation["annotations"], current_annotation["relations"])            
            
        else:
            raise ValueError('Incorrect variable type, expected a Pandas DF.')
            
               
    def write_relations(self, df, text_id, relation, overwrite = False):
        
        """
            write_relations
            Send relations data from the brat folder. Useful to pre-anotate some data.

            input :
                df, pd.Dataframe : dataframe containing relations data, should contains the text id, the relation name, the if of the linked annotations.
                text_id, str : name of the column in df which contains the document id
                relation, str : name of the column in df which contains the relation name
                overwrite, boolean : if True, the current annotation files are replaced by new data, otherwise, the new annotations are merged with existing one

                The other columns should contains the type_id of related entities, as outputed by the read_annotation method.
        """
        
        # Checking data types
        if (type(df) == type(pd.DataFrame())):
            
            # Loading df
            df = df.rename(columns = {text_id:"id",relation:"relation"})
            df["type_id"] = df.groupby("id").cumcount()+1 # type_id
            
            # Columns names
            old_columns = df.columns[np.isin(df.columns, ["id", "relation","type_id"]) == False]
            new_columns = "Arg"+np.array(list(range(1,len(old_columns)+1))).astype(str).astype(object)
            df = df.rename(columns = dict(zip(old_columns, new_columns)))
            
            # List of ids
            ids = df["id"].unique()

            # Loading current data
            current_annotation = self.read_annotation(ids)            
            current_relations = current_annotation["relations"]
            rmaxDFrelations = current_relations.set_index(["id"])["type_id"].str.slice(1,).astype(int).reset_index().groupby("id").max().rename(columns = {"type_id":"Rmax"})

            if (overwrite == True):
                df["type_id"] = "R"+df["type_id"].astype(str)
                new_relations = df
            else:
                df = df.join(rmaxDFrelations, on = "id").fillna(0)
                df["type_id"] = "R"+(df["type_id"]+df["Rmax"]).astype(int).astype(str)
                df = df.drop(columns = ["Rmax"])
                
                # Adding missing columns
                if (len(df.columns) > len(current_relations.columns)):
                    for column in df.columns[np.isin(df.columns, current_relations.columns) == False]:
                        current_relations[column] = np.nan
                else:
                    for column in current_relations.columns[np.isin(current_relations.columns, df.columns) == False]:
                        df[column] = np.nan
                
                new_relations = pd.concat((current_relations, df[current_relations.columns])).reset_index(drop = True)
                
            new_relations.drop_duplicates() ## Removing duplicates
            
            # Injecting new annotations
            current_annotation["relations"] = new_relations
            
            # Calling write function
            self._write_annotation(current_annotation["annotations"], current_annotation["relations"])
            
        else:
            raise ValueError('Incorrect variable type, expected a Pandas DF.')
            
    def _generate_annotations_str (self, annotations):
        
        annotations = annotations.reset_index(drop = True)
        annotations["label_span"] = annotations[["label","start","end"]].apply(lambda x: ' '.join(x.astype(str).values), axis = 1)
        annotations_str = '\n'.join(annotations[["type_id","label_span","word"]].apply(lambda x: '\t'.join(x.astype(str).values), axis = 1).values)

        return(annotations_str)
        
    def _generate_relations_str (self, relations):
            
        
        relations = relations.fillna('').applymap(lambda x: '' if x == 'nan' else x) #cleaning data
        columns = relations.columns[np.isin(relations.columns, ["id","type_id","relation"]) == False].values.tolist()
        boolmap = relations[columns].transpose().applymap(lambda x: int(x != ''))
        rct = relations[columns].transpose()

        temp_relations = (boolmap*(np.array(np.repeat(rct.index,rct.shape[1])).reshape(rct.shape)+':')+rct.astype(str)).transpose()

        relations_str = '\n'.join(relations[["type_id","relation"]].join(temp_relations[columns]).apply(lambda x: '\t'.join(x.values), axis = 1).values)

        return(relations_str)
    
    def _write_file(self, data):
        file = open(self.folder+str(data["id"])+".ann", "w")
        file.write(data["str_to_write"])
        file.close()
    
    def _write_annotation(self,annotations,relations):
        
        # Checking data types
        if (type(annotations) == type(pd.DataFrame()) and type(relations) == type(pd.DataFrame())):
            
            # Gerenating str
            data_annotations = annotations.groupby("id").agg(lambda x: self._generate_annotations_str(x)).iloc[:,0]
            data_relations = relations.groupby("id").agg(lambda x: self._generate_relations_str(x)).iloc[:,0]

            # Merging data
            data = pd.DataFrame({"annotations":data_annotations, "relations":data_relations}).fillna('')
            data["str_to_write"] = data.apply(lambda x : '\n'.join(x.values), axis = 1)
            data = data.reset_index().rename(columns = {"index":"id"})
            
            # Writting files
            data.apply(self._write_file, axis = 1)
            
            return(data)
            
        else:
            raise ValueError('Incorrect variable type, expected a Pandas DF.')

    def _export_conll_2003 (self, data):

        '''
            Internal function for export in conll format.
        '''

        # Creating i-label
        data["i-label"] = (data["label"] != "O").astype(int)*(data["i-type"]+'-')+data["label"]
        
        # Creating string
        data["str"] = data[["token","pos","chunks","i-label"]].apply(lambda x: ' '.join(x), axis = 1)
        
        connll_str = "-DOCSTART- -X- -X- O"+"\n\n"+"\n\n".join(
            data.groupby("id").agg(lambda x: "\n".join(x))["str"].values.tolist()
        )
        
        return(connll_str)

    def _get_tokenized_data(self, text_data, annotations_data, tokenizer = default_tokenizer, keep_empty = False):
        
        '''
            Internal function that process text and annotation data to calculate token, pos and chunks.
            
            Input :
                text_data : text data exported from current class
                annotations_data : annotations data exported from current class
                tokenizer : tokenizer function from extract_tools
                keep_empty : default False, parameter boolean, if True empty token are not removed, otherwise they are removed
            Output :
                Aggreged data in Pandas DataFrame.
        '''

        # Applying tokenizer to text
        text_data["tokens"] = text_data["text_data"].apply(tokenizer)

        # Exploding dataframe by tokens and rename column
        exploded_text_data = text_data[["id", "tokens"]].explode("tokens").reset_index(drop = True)
        exploded_text_data = exploded_text_data.join(
            exploded_text_data["tokens"] \
                .apply(pd.Series) \
                .rename(columns = {
                    0:'token',1:'start_offset',2:'end_offset', 3:'pos'
                })
        ) \
        .drop(columns = ["tokens"])
                
        # Getting entities from annotations
        
        ## We merge by offset
        
        ### Creating a word id and annotation id
        exploded_text_data = exploded_text_data \
                            .reset_index(drop = True) \
                            .reset_index() \
                            .rename(columns = {"index":"word_id"})
        
        annotations_data = annotations_data \
                            .reset_index() \
                            .rename(columns = {"index":"ann_id"})
        
        ### Offset of string
        text_offsets = pd.DataFrame(exploded_text_data[["id","word_id","start_offset","end_offset"]])

        text_offsets["start_offset"] = text_offsets["start_offset"].astype(int)
        text_offsets["end_offset"] = text_offsets["end_offset"].astype(int)
        text_offsets["offsets"] = text_offsets \
            .apply(
                lambda x: list(range(x["start_offset"], x["end_offset"]+1)), axis = 1
            )
        text_offsets = text_offsets[["id","word_id", "offsets"]] \
                        .explode("offsets")
        
        ### Offset of annotations
        
        ann_offsets = pd.DataFrame(
            annotations_data[["id", "ann_id", "start", "end"]]
        )
        ann_offsets["start"] = ann_offsets["start"].astype(int)
        ann_offsets["end"] = ann_offsets["end"].astype(int)

        if (ann_offsets.shape[0] > 0):
            ann_offsets["offsets"] = ann_offsets \
                .apply(
                    lambda x: list(range(x["start"], x["end"]+1)), axis = 1
                )
        else:
            ann_offsets["offsets"] = ''

        ann_offsets = ann_offsets[["id","ann_id", "offsets"]] \
                        .explode("offsets")
            
        # Merging by term
        
        text_offsets["uid"] = text_offsets["id"].astype(str) \
                            + text_offsets["offsets"].astype(str)
        
        ann_offsets["uid"] = ann_offsets["id"].astype(str) \
                            + ann_offsets["offsets"].astype(str)
        
        merged_id = text_offsets \
                    .join(
                        ann_offsets[["ann_id","uid"]].set_index("uid"), 
                        on = "uid"
                    ) \
                    .dropna()
        merged_id["ann_id"] = merged_id["ann_id"].astype(int)
        
        merged_id = merged_id[["word_id", "ann_id"]] \
            .set_index("ann_id") \
            .drop_duplicates() \
            .join(annotations_data, on = "ann_id")
        
        # Keeping last when duplicate word_id
        
        merged_id = merged_id \
            .drop_duplicates("word_id", keep = "last")
        
        # Joining annotation with word id

        output_df = exploded_text_data \
            .join(merged_id[["label","word_id"]] \
                .set_index("word_id"), 
                on = "word_id",
                how = "left") \
            .fillna("O")[["id", "token","label", "pos"]]
        
        # Creation of i-type : if O : i-type is B
        output_df["i-type"] = output_df \
            .groupby("id").agg(lambda x: ["B"]+["I"]*(len(x)-1))["label"].explode().reset_index()["label"]
        output_df.loc[output_df["label"] == "O", "i-type"] = 'B'
        
        # Empty chunks
        output_df["chunks"] = 'O'

        # Post - processing
        if (keep_empty == False):
            output_df = output_df[output_df["token"] != ''].reset_index(drop = True)
        
        return(output_df)

    def export(self, export_format = "conll-2003", tokenizer = default_tokenizer, keep_empty = False, entities = None):

        '''
            Function that generate an export file.
            Supported export format are :
                - conll-2003

            input :
                export_format : name of the export format
                tokenizer : tokenizer function from extract_tools
                keep_empty : default False, parameter boolean, if True empty token are not removed, otherwise they are removed
                entities : if None, all entities are send to the export file, if there is the conflict the most recent is used, otherwise the entities are selected before
            Output :
                str : output string in selected export format
        '''

        supported_export_format = {
            "conll-2003":self._export_conll_2003
        }
            
        # Check the export format
        if (export_format not in supported_export_format.keys()):
            raise Exception(str(export_format)+" format not supported. Export format should be one of these : {}".format(
            ", ".join(supported_export_format.keys())  
            ))
        
        # Create dataframe of tokenized word associated with annotations
        ## Getting data from brat
        text_data = self.read_text()
        annotations_data = self.read_annotation()["annotations"]

        ## Filtering entities
        if entities is not None:
            if type(entities) != type(list()):
                raise Exception("entities should be of type list")

            annotations_data = annotations_data[annotations_data["label"].isin(entities)] \
                                                                         .reset_index(drop = True)
        
        ## Parsing data
        data = self._get_tokenized_data(tokenizer=tokenizer, 
                                    text_data = text_data, 
                                    annotations_data = annotations_data, 
                                    keep_empty = keep_empty)

        # Execute the export format associated function
        data_str = supported_export_format[export_format](data = data)

        return(data_str)