# pandasToBrat

Ali BELLAMINE - contact@alibellamine.me
_Last version : 1.0 - 28/10/2020_ 

Main repository : https://gogs.alibellamine.me/alibell/pandasToBrat/

## What is pandasToBrat ?

pandasToBrat is a library to manage brat configuration and brat data from a Python interface.

### What can it do ?

- Reading brat annotations and relations configuration to python dictionnary
- Writting brat annotations and relations configuration from python dictionnary
- Reading brat text data to python pandas dataframe
- Writting brat text file from python pandas Series
- Reading brat annotations and relations 
- Writting brat annotations and relations from python pandas DataFrame
- Export data to ConLL-2003 format

### What it doesn't support ?

- Keyboard shortcut configuration
- Event, Attribution, Modification, Normalization and Notes annotations
- Relation type in relations configuration

## How to use it ?

### Installation

Clone the current repository :
```
    git clone https://gogs.alibellamine.me/alibell/pandasToBrat
```

Install dependencies with pip.

```
    pip install -r requirements.txt
```

Then install the library :

```
    pip install -e .
```

### Loading a brat folder

Instantiate the brat library with the folder path :

```
    from pandasToBrat import pandasToBrat

    brat_data = pandasToBrat(FOLDER_PATH)
```

### Parameters

Parameters are stored in a dictionnary :

```
    {
        "entities":ENTITIES_CONFIGURATION_DATA,
        "relations":RELATIONS_CONFIGURATION_DATA
    }
```

#### Entities configuration data

Dictionnary formated as :
```
    {
        LABEL_NAME:{
            LABEL_NAME_CHILD1:True,
            LABEL_NAME_CHILD2:True,
            LABEL_NAME_CHILD3:{
                LABEL_NAME_CHILD3_CHILD1:True
            }
        }
    }
```

Each entry is an entitie.
An entitie can either be setted as True, it have no child, or have on or many childrens in which case is contains a dictionnary.

#### Relations configuration data

Dictionnary formated as :
```
    {
        RELATION_NAME:{
            "args":[ENTITIES_NAME,...]
        }
    }
```

Each entrie of the dictionnary is a relation.
Each relation have a relation name and defined with a sub-dictionnary containing an args entrie.
The args entrie contains a list of entities that are concerned by the relation.

####  Read and write parameters

##### Getting parameters

You can read the current parameters using the dedicated method :

```
    bratData.read_conf()
```

##### Writtings parameters

You can write parameters using the dedicated method :

```
    bratData.write_conf(entities = ENTITIES_CONFIGURATION, relations = RELATIONS_CONFIGURATION)
```

The ENTITIES_CONFIGATION is a dictionnary formated as described in the "Entities configuration data" chapter.

The RELATIONS_CONFIGURATION is a dictionnary formated as described in the "Relations configuration data" chapter.

### Text

Text is stored in a Pandas Dataframe with two columns :
- id : document id, which is contained in the .txt filename
- text_data : document data

#### Read and write text

##### Getting text data

```
    bratData.read_text()
```

#### Sending text data

```
    bratData.write_text(text_id=TEXT_ID_SERIES, text = TEXT_SERIES, empty = EMPTY_PARAMETER, overWriteAnnotations = OVERWRITE_ANNOTATIONS_PARAMETERS)
```

The required parameters are text_id and text which are Pandas Series, which should be of the same size containing for the first one the document unique id and the second one the document text data.

The empty parameters is used to empty the current folder. If set as True, the Brat folder is emptied of all text and annotations data. Configuration is not erased.

The overwrite annotations parameter is used to overwrite the current annotation (.ann) file with an empty one, it is useful if you want to remove the existing annotations when you are modifiying a text file.

This way, you can :
- Overwrite all data with empty set as True
- Only overwritting new data with empty set as False and overWriteAnnotations set as True : you write new file, if the id already exist it is overwritten, if not is it ignored.

### Annotations

Parameters are stored in a dictionnary :

```
    {
        "annotations":ANNOTATIONS_ANNOTATIONS,
        "relations":RELATIONS_ANNOTATIONS
    }
```

#### Annotations format

Annotations are word labeled with entities.

It is formatted as a Pandas DataFrame, containing the following columns :
- id : Document id, one document can have mutiples annotations
- type_id : annotation number inside the same document, from T1 to Tn, with n the number of annotated string, it is used to match annotations with relations
- word : the annotated string
- label : the entitie related to the annotated string
- start : the annotated string start offset
- end : the annotated string end offset

#### Relations format

Annotations are relations between annotations.

It is formatted as a Pandas DataFrame, containing the following columns :
- id : Document id, one document can have mutiples relations
- type_id : relation number inside the same document, from R1 to Rn, with n the number of relations
- relation : The relation Name
- ArgX : The annotated entitie which a linked by the relation, each column refer to an entitie, the entitie id correspond to the annotations DataFrame "type_id" column

#### Read and write annotations


##### Getting annotations data

```
    bratData.read_annotation()
```

#### Sending annotations data


##### Write annotations subpart of annotations

```
    bratData.write_annotations(df, text_id, word, label, start, end, overwrite=OVERWRITE_OPTION)
```

The first parameter is the datafame containing the annotations.
It should be formated as described in the "Annotations format" subpart.

The text_id, word, label, start and end are the name of the column inside the dataframe which contains the related data.

The overwrite option can be set as True to overwrite existing annotations, otherwise the dataframe's data are added to existing annotations data.

##### Write relations subpart of annotations

```
    bratData.write_relations(df, relation, overwrite=OVERWRITE_OPTION)
```

The first parameter is the datafame containing the relations.
It should be formated as described in the "Relations format" subpart.

The text_id and relation are the name of the column inside the dataframe which contains the related data.
The other columns should contains the type_id of related entities, as outputed by the read_annotation method.

The overwrite option can be set as True to overwrite existing annotations, otherwise the dataframe's data are added to existing annotations data.

### Export data to standard format

The only currently supported format is ConLL-2003.

To export data, you can use the export method.

```
    bratData.export(export_format = EXPORT_FORMAT, tokenizer = TOKENIZER, entities = ENTITIES_OPTION, keep_empty = KEEP_EMPTY_OPTION)
```

The export_format parameter is used to specify the export format. The only one, which is the default one, supported is ConLL-2003.
The tokenizer parameter contains the tokenizer functions. Tokenizers functions are stored in pandasToBrat.extract_tools. The aim of the function is to generate tokens and pos tag from text. The default one, _default_tokenizer_, is the simplest one, that split on space and new line character.
You can also use Spacy tokenizer, in that case you should import the spacy_tokenizer functions as demonstrated in this example :

```
    from pandasToBrat.extract_tools import spacy_tokenizer
    import spacy

    nlp = spacy.load(SPACY_MODEL)
    spacy_tokenizer_loaded = spacy_tokenizer(nlp)

    bratData.export(tokenizer = spacy_tokenizer_loaded)
```

You can restrict the export to a limited set of entities. For that, the list of entities are specified in the entities parameter. If set as None, which is the default value, all entities will we considered. If a word contains many entities, the last one is kept.

Finally, the keep_empty option is defaultly set as False. This means that every empty tokens will be removed from the exported data.
You can set it as True if you want to keep empty tokens.
