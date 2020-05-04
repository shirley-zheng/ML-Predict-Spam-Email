# Objective: Load and process a Ham vs Spam data set using the Naive Bayes classifier.
# Note: This data set has already been preprocessed for you by removing punctuation, removing stop words, and stemming. 

#Create the readDirectory() method:
readDirectory <- function(dirname) {
  #Store emails in a list
  emails = list();
  #Get list of filenames in the directory
  filenames = dir(dirname, full.names = TRUE);
  for (i in 1:length(filenames)){
    emails[[i]] = scan(filenames[i], what = "", quiet = TRUE);
  }
  return(emails)
}

#Load data 
ham_test   <- readDirectory("~/Desktop/SpamHam/ham-test")
spam_test  <- readDirectory("~/Desktop/SpamHam/spam-test")
ham_train  <- readDirectory("~/Desktop/SpamHam/ham-train")
spam_train <- readDirectory("~/Desktop/SpamHam/spam-train")


#Create a dictionary 
sorted_dictionary <- function(emails) {
  # combine different lists of emails
  dictionary_full <- unlist(emails)
  #T abulates the full dictionary
  tabulate_dic <- tabulate(factor(dictionary_full))
  # Find unique values
  dictionary <- unique(dictionary_full)
  # Sort them alphabetically
  dictionary <- sort(dictionary)
  dictionary_df <- data.frame(word = dictionary, count = tabulate_dic)
  sort_dictionary <- dictionary_df[order(dictionary_df$count, decreasing = TRUE),];
  return(sort_dictionary)
}


all_emails <- c(ham_train, ham_test, spam_train, spam_test)
dictionary <- sorted_dictionary(all_emails)
head(dictionary)

#word count
#6776    ect  2681
#10104   hou  1309
#7190  enron   691
#15815 pleas   672
#4442    com   615
#5565   deal   563


#Create document term matrix (DTM)
#Make a function to create a document term matrix for any given list of emails.
document_term_matrix <- function(emails, dictionary_words) {
dtm <- sapply(seq_along(emails), 
function(i) dictionary_words %in% emails[[i]] )
return (dtm)
}

#Create DTM's for each of the four email groups.
make.document.term.matrix <- function(emails,dictionary){
  # This takes the email and dictionary objects from above and outputs a
  # document term matrix
  num.emails <- length(emails);
  num.words <- length(dictionary$word);
  # Instantiate a matrix where rows are documents and columns are words
  dtm <- mat.or.vec(num.emails,num.words); # A matrix filled with zeros
  for (i in 1:num.emails){
    num.words.email <- length(emails[[i]]);
    email.temp <- emails[[i]];
    for (j in 1:num.words.email){
      ind <- which(dictionary$word == email.temp[j]);
      dtm[i,ind] <- dtm[i,ind] + 1;
    }
  }
  return(dtm);
}

dtm_ham_train  <- t(document_term_matrix(ham_train,  dictionary$word))
dtm_ham_test   <- t(document_term_matrix(ham_test,   dictionary$word))
dtm_spam_train <- t(document_term_matrix(spam_train, dictionary$word))
dtm_spam_test  <- t(document_term_matrix(spam_test,  dictionary$word))


#Create a naive Bayes classifier
#Setup a Naive Bayes classifier to predict whether the emails are spam.
log_pvec <- function(dtm, mu) {
  pvec_no_mu <- colSums(dtm) # sum instances of each word
  n_words    <- sum(pvec_no_mu) # sum number of words
  dic_len    <- length(pvec_no_mu) # dictionary size
  log_pvec   <- log(pvec_no_mu + mu) - 
    log(mu * dic_len + n_words) # incorporate mu and normalize
  return(log_pvec)
}

mu = 1/length(dictionary$word)
log.pvec.spam <- log_pvec(dtm_spam_train, mu)
log.pvec.ham <- log_pvec(dtm_ham_train, mu)
log.pvec.spam <- log_pvec(dtm_spam_train, mu)
log.prior.spam <- log(0.5)
log.prior.ham <- log(0.5)


naive.bayes <- function(log.pvec.ham, log.pvec.spam,log.prior.ham,log.prior.spam,dtm_test){
  p_spam <- log.pvec.spam %*% t(dtm_test) + log.prior.spam
  p_ham <- log.pvec.ham %*% t(dtm_test) + log.prior.ham
  result <- ifelse(p_spam>p_ham, "spam", "ham")
  return (result)
}

#dictionary[1:10,]
index <- as.numeric(rownames(dictionary[1:10,]))

# Find total count of corresponding indexes in each dtm
colSums(dtm_ham_train)[index] 
#[1] 0 1 0 0 0 0 0 0 0 0
colSums(dtm_spam_train)[index]
#[1] 2 0 2 1 2 1 1 1 1 0

# Predict using the naive.bayes function
pred_spam <- naive.bayes(log.pvec.ham, log.pvec.spam,log.prior.ham,log.prior.spam,dtm_spam_test)
pred_ham <- naive.bayes(log.pvec.ham, log.pvec.spam,log.prior.ham,log.prior.spam,dtm_ham_test)

#display the outcome
table(pred_spam) 
#pred_spam
#ham spam 
#14  136 
table(pred_ham)
#pred_ham
#ham spam 
#145    5 

#Summary: 
#The number of correct SPAM classifications = 136; incorrect HAM classifications = 14
#The number of correct HAM classifications = 145; incorrect HAM classifications = 5



