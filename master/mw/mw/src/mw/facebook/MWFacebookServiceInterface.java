package mw.facebook;


public interface MWFacebookServiceInterface {

	public String[] searchIDs(String name);
	public String getName(String id) throws MWUnknownIDException;
	public String[] getFriends(String id) throws MWUnknownIDException;
	public String[][] getFriendsBatch(String[] ids) throws MWUnknownIDException;
	public int getServerStatus(int interval);

}
