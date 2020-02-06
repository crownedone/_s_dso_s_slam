#include "Input.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <boost/filesystem.hpp>

using namespace std;

bool RGBD_TUM::read(const std::string& associationsFilename_, const std::string& sequenceFolder_)
{
    printf("Opening %s\n", associationsFilename_.c_str());

    if (!boost::filesystem::exists(associationsFilename_) ||
        !boost::filesystem::is_regular_file(associationsFilename_))
    {
        printf("Provided associations file is invalid %s", associationsFilename_.c_str());
        return false;
    }

    if(!sequenceFolder_.empty() && (!boost::filesystem::exists(sequenceFolder_) ||
                                    boost::filesystem::is_regular_file(sequenceFolder_)))
    {
        printf("Provided sequence folder is invalid %s", sequenceFolder_.c_str());
        return false;
    }

    associationsFilename = associationsFilename_;
    sequenceFolder =  sequenceFolder_;
    vstrImageFilenames.resize(0);
    vstrImageFilenames1.resize(0);
    vTimestamps.resize(0);

    ifstream fAssociation;
    fAssociation.open(associationsFilename.c_str());

    while (!fAssociation.eof())
    {
        string s;
        getline(fAssociation, s);

        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenames1.push_back(sD);
        }
    }

    if (vstrImageFilenames.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return false;
    }
    else if (vstrImageFilenames1.size() != vstrImageFilenames.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return false;
    }
    else if(vstrImageFilenames1.size() != vTimestamps.size())
    {
        cerr << endl << "Different number of images from timestamps." << endl;
        return false;
    }

    printf("Read done %s\n", associationsFilename_.c_str());
    return true;
}