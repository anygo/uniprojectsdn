#include <string>
#include <vector>
#include <blackjack/core/Algorithm.h>
#include <blackjack/core/Card.h>
#include <blackjack/core/RegisteredPlayer.h>
#include <blackjack/core/Types.h>
#include <map>
#include <iostream>
#include <ctime>
#include <cstdlib>


namespace blackjack {

class Neumann : public RegisteredPlayer<Neumann>
{
 private:
   typedef RegisteredPlayer<Neumann>    		     Base;
   typedef std::vector<CardID>          		 	 Cards;
   typedef std::map<PlayerID, std::vector<CardID> >  Others;
   typedef std::pair<PlayerID, std::vector<CardID> > OthersCardsPair;

 public:
   Neumann();
   ~Neumann();

   virtual bool offer();
   virtual void give( CardID card );
   virtual void newRound( bool shuffle );
   virtual void newGame();
   virtual void introduce(PlayerID player);
   virtual void playerDrawsCard(PlayerID player, CardID card);

 private:
   Cards       cards_;
   value_t     value_;
   Others	   others_;
   int 		   usedTwos_;
   static bool registered_;
};


bool Neumann::registered_ = Neumann::registerThis();


Neumann::Neumann()
   : Base( "Dominik Neumann" )
   , cards_()
   , value_()
   , others_()
{}


Neumann::~Neumann()
{}


void Neumann::playerDrawsCard(PlayerID player, CardID card) {
	others_[player].push_back(card);
	if (card->type() == two) ++usedTwos_;
}


void Neumann::introduce(PlayerID player) {
	others_.insert(OthersCardsPair(player, std::vector<CardID>()));
}


bool Neumann::offer()
{
	
   value_t cur_max = 0;
   for (Others::iterator it = others_.begin(); it != others_.end(); ++it)
   {
   		value_t tmp_value = count((*it).second.begin(), (*it).second.end());
		if (tmp_value > cur_max && tmp_value <= 21U) {
			cur_max = tmp_value;
		}
   }

   // "strategy"...
   if (value_ >= 20U) return false;
   else if (value_ <= 16U) return true;
   else if (value_ == 17U) {
		if ((rand() % 10) > 1) return true;
		else return false;
   }
   else if (value_ == 18U) {
		if ((rand() % 10) > 3) return true;
		else return false;
   }
   else if (value_ < cur_max) {
		if (usedTwos_ >= 19) return false;
		else return true;
   }
   else return false;
}


void Neumann::give( CardID card )
{
	if (card->type() == two) ++usedTwos_;
   cards_.push_back(card);
   value_ = count(cards_.begin(), cards_.end());
}


void Neumann::newRound(bool shuffle)
{
   if (shuffle) {
		usedTwos_ = 0;
   }

   value_ = 0;
   cards_.clear();
   for (Others::iterator it = others_.begin(); it != others_.end(); ++it) {
   		(*it).second.clear();
   }
}

void Neumann::newGame() {
    srand(time(0));
	others_.clear();
}


} // blackjack
