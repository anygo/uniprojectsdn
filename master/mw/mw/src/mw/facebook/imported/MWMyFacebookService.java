
package mw.facebook.imported;

import javax.jws.WebMethod;
import javax.jws.WebParam;
import javax.jws.WebResult;
import javax.jws.WebService;
import javax.jws.soap.SOAPBinding;
import javax.xml.bind.annotation.XmlSeeAlso;


/**
 * This class was generated by the JAX-WS RI.
 * JAX-WS RI 2.1.6 in JDK 6
 * Generated source version: 2.1
 * 
 */
@WebService(name = "MWMyFacebookService", targetNamespace = "http://facebook.mw/")
@SOAPBinding(style = SOAPBinding.Style.RPC)
@XmlSeeAlso({
    ObjectFactory.class
})
public interface MWMyFacebookService {


    /**
     * 
     * @param arg0
     * @return
     *     returns java.lang.String
     * @throws MWUnknownIDException_Exception
     */
    @WebMethod
    @WebResult(partName = "return")
    public String getName(
        @WebParam(name = "arg0", partName = "arg0")
        String arg0)
        throws MWUnknownIDException_Exception
    ;

    /**
     * 
     * @param arg0
     * @return
     *     returns mw.facebook.imported.StringArray
     * @throws MWUnknownIDException_Exception
     */
    @WebMethod
    @WebResult(partName = "return")
    public StringArray getFriends(
        @WebParam(name = "arg0", partName = "arg0")
        String arg0)
        throws MWUnknownIDException_Exception
    ;

    /**
     * 
     * @param arg0
     * @return
     *     returns mw.facebook.imported.StringArrayArray
     * @throws MWUnknownIDException_Exception
     */
    @WebMethod
    @WebResult(partName = "return")
    public StringArrayArray getFriendsBatch(
        @WebParam(name = "arg0", partName = "arg0")
        StringArray arg0)
        throws MWUnknownIDException_Exception
    ;

    /**
     * 
     * @param arg0
     * @return
     *     returns mw.facebook.imported.StringArray
     */
    @WebMethod
    @WebResult(partName = "return")
    public StringArray searchIDs(
        @WebParam(name = "arg0", partName = "arg0")
        String arg0);

    /**
     * 
     * @param arg0
     * @return
     *     returns int
     */
    @WebMethod
    @WebResult(partName = "return")
    public int getServerStatus(
        @WebParam(name = "arg0", partName = "arg0")
        int arg0);

}
