//=================================================================================================
/*
//  Advanced Programming Techniques (AdvPT)
//  Winter Term 2009/2010
//  Project phase - Black Jack
//
//  Copyright (C) 2010 Dominik Neumann
*/
//=================================================================================================


#include <string>
#include <vector>
#include <blackjack/core/Algorithm.h>
#include <blackjack/core/Card.h>
#include <blackjack/core/RegisteredPlayer.h>
#include <blackjack/core/Types.h>


namespace blackjack {

class Neumann : public RegisteredPlayer<Neumann>
{
 private:
   typedef RegisteredPlayer<Neumann>    Base;
   typedef std::vector<CardID>          Cards;

 public:
   Neumann();
   ~Neumann();

   virtual bool offer();
   virtual void give( CardID card );
   virtual void newRound( bool shuffle );

 private:
   Cards       cards_;
   value_t     value_;
   static bool registered_;
};


bool Neumann::registered_ = Neumann::registerThis();


Neumann::Neumann()
   : Base( "Dominik Neumann" )
   , cards_()
   , value_()
{}


Neumann::~Neumann()
{}


bool Neumann::offer()
{
   if( value_ > 16U ) return false;
   else               return true;
}


void Neumann::give( CardID card )
{
   cards_.push_back( card );
   value_ = count( cards_.begin(), cards_.end() );
}


void Neumann::newRound( bool /*shuffle*/ )
{
   value_ = 0;
   cards_.clear();
}


} // blackjack
