#include "ICP.txx"
#include "RBC.txx"
#include "VolumeManager.txx"


template class VolumeManager<128, 40>;

//template class RBC<512,6>;
//template class RBC<1024,6>;
//template class RBC<2048,6>;
//template class RBC<4096,6>;
//template class RBC<8192,6>;
//template class RBC<16384,6>;

template class ICP<512,6>;
template class ICP<1024,6>;
template class ICP<2048,6>;
template class ICP<4096,6>;
template class ICP<8192,6>;
template class ICP<16384,6>;