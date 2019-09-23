init 999 python:
    #The classifier we'll uses
    classifier = store.mas_nlp.NaiveBayesTextClassifier()

    #Cleaned gamedir, for ease of use
    clean_gamedir = renpy.config.gamedir.replace("\\", "/")

    def mas_trainClassifier(classifier=None, trainingfile=None):
        """
        Trains the classifier using the training file provided

        IN:
            - classifier:
                The classifier to train

            - trainingfile:
                the file to read training data from. If none, we assume the one at
                mod_assets/nlp_utils/trainingdata.txt
        """
        sentences = list()
        classes = list()

        if not trainingfile:
            trainingfile = clean_gamedir + "/mod_assets/nlp_utils/trainingdata.txt"

        #If file isn't loadable, return
        if not renpy.loadable(trainingfile):
            return

        #Otherwise, get Training Data
        with renpy.file(trainingfile) as trn_data:
            for line in trn_data:
                line = line.strip()

                #If we have nothing there, or the line's 'commented', skip
                if line == '' or line[0] == '#': continue

                #Split line
                x = line.split('\\')

                classes.append(x[0])
                sentences.append(mas_nlp.pre_process_sentence(x[1]).split())

        #Train the classifier
        classifier.train(sentences,classes)


    #Set up Morphy/POS Tagger
    morphy = store.mas_nlp.Morphy(base_dir=clean_gamedir + "/mod_assets/nlp_utils/pickles/")
    #TODO: implement tokenize
    pos_tagger = store.mas_nlp.PerceptronTagger(base_dir=clean_gamedir + "/mod_assets/nlp_utils/")


    def mas_classifyStr(classify_str, classifier=None):
        """
        Classifies the string using nlp_utils, and the classifier provided

        IN:
            - classify_str:
                The string to classify

            - classifier:
                The classifier to use
                NOTE: If no classifier provided, this just returns None.
        """
        if not classifier or not classify_str:
            return

        return classifier.classify(store.mas_nlp.pre_process_sentence(classify_str).split())