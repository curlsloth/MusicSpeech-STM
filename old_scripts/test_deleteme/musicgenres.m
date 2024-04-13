close all; clear all; clc;

CDfile = '/Users/andrewchang/NYU_research/MusicSpeech-STM/data/musicCorp/CD/CD_music_list.xlsx';
fmafile = '/Users/andrewchang/NYU_research/MusicSpeech-STM/data/musicCorp/fma_large/fma_metadata/genres.csv';
OS55file = '/Users/andrewchang/NYU_research/MusicSpeech-STM/data/musicCorp/MagnaTagATune/annotations_final.csv';
%%
% OS5-5
OS55full = readtable(OS55file);
OS55_genreOrg = OS55full.Properties.VariableNames;
genre_labs = {'hardRock', 'world', 'clasical', 'chorus','femaleOpera','heavy', 'funky', 'foreign', ...
    'classical', 'softRock', 'jazz', 'electric', 'folk', 'newAge', 'synth', 'funk', 'middleEastern', ...
    'medieval', 'classic', 'electronic', 'choral', 'opera', 'country', 'electro', 'reggae', 'tribal', ...
    'irish', 'electronica', 'operatic', 'arabic', 'trance', 'chant', 'drone', ...
    'synthesizer', 'heavyMetal', 'disco', 'industrial', 'jungle', 'pop', 'celtic', ...
    'eastern', 'blues', 'rock', 'dance', 'jazzy', 'techno', 'monks', 'oriental', 'choir', 'rap', ...
    'metal', 'hipHop', 'baroque', 'india', 'notEnglish', 'ambient'};
OS55_genres = OS55full(:,find(ismember(OS55_genreOrg, genre_labs)));
OS55Genres = {};
for i = 1:size(OS55_genres,1)
    OS55Genres{i,1} = OS55_genres.Properties.VariableNames(logical(OS55_genres{i,:}));
    OS55Genres_readable{i,1} = strjoin(OS55Genres{i,1}, ',');
end
OS55_genres.genres = OS55Genres;
OS55_genres.genresLabs = OS55Genres_readable;
OS55_genresSum = OS55_genres{:,{'genres', 'genresLabs'}};

classical = {'clasical', 'classical', 'classic', 'baroque'};
electronic = {'electric', 'electronic', 'electronica', 'electro', 'techno', 'industrial'};
world = {'world', 'foreign', 'middleEastern', 'reggae', 'tribal', 'celtic', 'irish', 'arabic', 'eastern', 'oriental', 'india', 'notEnglish', 'medieval'};
funk = {'funky', 'funk', 'jungle'};
synth = {'synth', 'synthesizer'};
rock = {'hardRock', 'softRock', 'rock'};
jazz_blues = {'jazz', 'jazzy', 'blues'};
metal = {'heavyMetal', 'metal', 'heavy'};
dance = {'dance', 'disco', 'trance'};
choral = {'choral', 'choir', 'chorus', 'chant', 'monks'}; % choral is not a style.
opera = {'opera', 'operatic', 'femaleOpera'};
rap_hipHop = {'rap', 'hipHop'};
pop = {'pop', 'ambient', 'newAge', 'drone'};
folk_country = {'folk', 'country'};

GenresLabs = {[classical, opera]; [electronic, synth, dance]; [world, choral]; [funk,jazz_blues]; rap_hipHop; [rock, metal]; pop; folk_country};
GenresRename = {'classical'; 'electronic'; 'world_international';'jazz_blues_funk'; 'rap_hipHop'; 'rock_metal_punk'; 'pop';'folk_country'};

for ii = 1:size(OS55_genresSum,1)
    if ~isempty(OS55_genresSum(ii, 1))
        curLabs = OS55_genresSum{ii,1};
        for g = 1:length(GenresLabs)
            if sum(ismember(curLabs, GenresLabs{g,1}))~=0
                curLabs(ismember(curLabs, GenresLabs{g,1})) = GenresRename(g,1);
            end
        end
        OS55_genresSumNew{ii,1} = unique(curLabs);
        OS55_sumRead{ii,1} = strjoin(OS55_genresSumNew{ii,1}, ',');
        clear curLabs;
    end
end
%%
OS55_genresSum1 = OS55_genresSumNew(~cellfun(@isempty, OS55_genresSumNew));
for gg = 1:length(OS55_genresSum1)
    if sum(ismember(OS55_genresSum1{gg,1}, {'electronic'}))~=0 & length(OS55_genresSum1{gg,1}) > 1
        curGenre = OS55_genresSum1{gg,1};
        curGenre(ismember(curGenre, {'electronic'})) = []; 
        OS55_genresSum1{gg,1} = curGenre;
    end
    if sum(ismember(OS55_genresSum1{gg,1}, {'world_international'}))~=0 & length(OS55_genresSum1{gg,1}) > 1
        curGenre = OS55_genresSum1{gg,1};
        curGenre(ismember(curGenre, {'world_international'})) = []; 
        OS55_genresSum1{gg,1} = curGenre;
    end
    if sum(ismember(OS55_genresSum1{gg,1}, {'classical'}))~=0
        curGenre = OS55_genresSum1{gg,1};
        curGenre(~ismember(curGenre, {'classical'})) = []; 
        OS55_genresSum1{gg,1} = curGenre;
    end
    if sum(ismember(OS55_genresSum1{gg,1}, {'rap_hipHop'}))~=0
        curGenre = OS55_genresSum1{gg,1};
        curGenre(~ismember(curGenre, {'rap_hipHop'})) = []; 
        OS55_genresSum1{gg,1} = curGenre;
    end
    if sum(ismember(OS55_genresSum1{gg,1}, {'pop'}))~=0
        curGenre = OS55_genresSum1{gg,1};
        curGenre(~ismember(curGenre, {'pop'})) = []; 
        OS55_genresSum1{gg,1} = curGenre;
    end
    if sum(ismember(OS55_genresSum1{gg,1}, {'jazz_blues_funk'}))~=0
        curGenre = OS55_genresSum1{gg,1};
        curGenre(~ismember(curGenre, {'jazz_blues_funk'})) = []; 
        OS55_genresSum1{gg,1} = curGenre;
    end
    if sum(ismember(OS55_genresSum1{gg,1}, {'folk_country'}))~=0
        curGenre = OS55_genresSum1{gg,1};
        curGenre(~ismember(curGenre, {'folk_country'})) = []; 
        OS55_genresSum1{gg,1} = curGenre;
    end
    OS55_sumRead1{gg,1} = strjoin(OS55_genresSum1{gg,1}, ',');
end

for i = 1:size(GenresRename,1)
    tracks = length(OS55_sumRead1(ismember(OS55_sumRead1,GenresRename(i))));
    OS55_genresCount{i} = tracks;
end
OS55_genresCount = cell2table(OS55_genresCount');
OS55_genresCount.Properties.VariableNames = {'nTracks'};
OS55_genresCount.genres = GenresRename;
OS55_genresCount = sortrows(OS55_genresCount, 'nTracks', 'descend');
disp(OS55_genresCount);

% nTracks            genres         
% _______    _______________________
% 
%  5224      {'classical'          }
%  3054      {'pop'                }
%  2905      {'electronic'         }
%  1897      {'rock_metal_punk'    }
%  1455      {'world_international'}
%   757      {'jazz_blues_funk'    }
%   413      {'folk_country'       }
%   112      {'rap_hipHop'         }

%% Other corpora
% fma
fma_large = readtable(fmafile);
fma_genres = unique(fma_large(fma_large{:,'parent'}==0, {'top_level', 'title', 'x_tracks'}), 'stable');
fma_genres = sortrows(fma_genres, 'x_tracks', 'descend');
disp(fma_genres);
% top_level             title             x_tracks
% _________    _______________________    ________
% 
%     38       {'Experimental'       }     38154  
%     15       {'Electronic'         }     34413  
%     12       {'Rock'               }     32923  
%   1235       {'Instrumental'       }     14938  
%     10       {'Pop'                }     13845  
%     17       {'Folk'               }     12706  
%     21       {'Hip-Hop'            }      8389  
%      2       {'International'      }      5271  
%      4       {'Jazz'               }      4126  
%      5       {'Classical'          }      4106  
%      9       {'Country'            }      1987  
%     20       {'Spoken'             }      1876  
%      3       {'Blues'              }      1752  
%     14       {'Soul-RnB'           }      1499  
%      8       {'Old-Time / Historic'}       868  
%     13       {'Easy Listening'     }       730  

% ISMIR04: 
% The training and development set each consist of:
% classical: 320 files
% electronic: 115 files
% jazz_blues: 26 files
% metal_punk: 45 files
% rock_pop: 101 files
% world: 122 files 
% 
% The evaluation set consists of 729 tracks with a similar distribution. 

% Homburg: total 1886
% Blues 120 
% Electronic 113 
% Jazz 319 
% Pop 116 
% Rap/HipHop 300 
% Rock 504 
% Folk/Country 222 
% Alternative 145 
% Funk/Soul 47 

% CD
CDs = readtable(CDfile);
CDs.Properties.VariableNames = {'artist','album','piece','duration','genre','instrument'};
cd_genres = unique(CDs(:,'genre'), 'stable');
for i = 1:size(cd_genres,1)
    cur_genre = cd_genres{i,'genre'};
    tracks = length(CDs{strcmp(CDs{:,'genre'},cur_genre),'genre'});
    cd_genres.nTracks{i} = tracks;
end
cd_genres = sortrows(cd_genres, 'nTracks', 'descend');
disp(cd_genres);
%       genre           nTracks
% __________________    _______
% 
% {'Classical'     }    {[355]}
% {'Rock'          }    {[ 99]}
% {'Spoken & Audio'}    {[ 28]}
% {'Jazz'          }    {[ 26]}
% {'Pop'           }    {[ 18]}
