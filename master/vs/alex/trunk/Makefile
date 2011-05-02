all: marshalling-test proxy-test

marshalling-test:
	javac -cp /usr/share/java/junit.jar:vsue/marshalling: vsue/tests/*.java
	junit vsue.tests.VSMarshallingTest
	junit vsue.tests.VSExternalizableTest

proxy-test:
	javac -cp /usr/share/java/junit.jar:vsue/proxy: vsue/tests/*.java
	junit vsue.tests.VSProxyTest

clean:
	rm -f vsue/*/*.class

.PHONY: all clean proxy-test marshalling-test

