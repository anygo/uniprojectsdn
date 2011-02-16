BIN_PATH:=./bin
SRC:=./mw/facebook/MWFacebookServer.java

FACEBOOK_NAME:=MWFacebookService
MW_SERVER_URL:=http://`hostname --fqdn`:18081/${FACEBOOK_NAME}?wsdl

all:
	@mkdir -p ${BIN_PATH}
	@javac -d ${BIN_PATH} ${SRC}

start-facebook-service:
	@LANG="en_US.UTF-8" java -cp ${BIN_PATH} mw.facebook.MWFacebookServer ${MW_SERVER_URL}
