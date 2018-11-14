IGNORE_LABEL = -1
UNKNOWN_LABEL = 0
START_TOKEN = '<s>'
END_TOKEN = '</s>'
UNKNOWN_TOKEN = '<unk>'

# http://www.optasports.com/praxis/documentation/football-feed-appendices/f24-appendices.aspx

event_type_mapper = {
    1: 'pass', 2: 'offside pass', 3: 'take on', 4: 'foul', 5: 'out',
    6: 'corner awarded', 7: 'tackle', 8: 'interception', 9: 'turnover',
    10: 'save', 11: 'claim', 12: 'clearance', 13: 'miss', 14: 'pos',
    15: 'attempt saved', 16: 'goal', 17: 'card', 18: 'player off',
    19: 'player on', 20: 'player retired', 21: 'player returns',
    22: 'player becomes goalkeeper', 23: 'goalkeeper becomes player',
    24: 'condition change', 25: 'official change', 27: 'start delay',
    28: 'end delay', 30: 'end', 32: 'start', 34: 'team set up',
    35: 'player changed position', 36: 'player changed Jersey number',
    37: 'collection end', 38: 'temp_goal', 39: 'temp_attempt',
    40: 'formation change', 41: 'punch', 42: 'good skill',
    43: 'deleted event', 44: 'aerial', 45: 'challenge', 47: 'rescinded card',
    49: 'ball recovery', 50: 'dispossessed', 51: 'error', 52: 'keeper pick-up',
    53: 'cross not claimed', 54: 'smother', 55: 'offside provoked',
    56: 'shield ball opp', 57: 'foul throw-in', 58: 'penalty faced',
    59: 'keeper sweeper', 60: 'chance missed', 61: 'ball touch',
    63: 'temp_save', 64: 'resume', 65: 'contentious referee decision',
    66: 'possession data', 67: '50/50', 68: 'referee drop ball',
    69: 'failed to block', 70: 'injury time announcement', 71: 'coach setup',
    72: 'caught offside', 73: 'other ball contact', 74: 'blocked pass',
    75: 'delayed start', 76: 'early end', 77: 'player off pitch'
}

qualifier_type_mapper = {
    1: 'long ball', 2: 'cross', 3: 'head pass', 4: 'through ball',
    5: 'free kick taken', 6: 'corner taken', 7: 'players caught offside',
    8: 'goal disallowed', 106: 'attacking pass', 107: 'throw-in',
    140: 'pass end x', 141: 'pass end y', 152: 'direct', 155: 'chipped',
    156: 'lay-off', 157: 'launch', 168: 'flick-on', 193: 'goal measure',
    195: 'pull back', 196: 'switch of play', 210: 'assist', 212: 'length',
    213: 'angle', 218: '2nd assist', 219: 'players on both posts',
    220: 'player on near post', 221: 'player on far post',
    222: 'no players on posts', 223: 'in-swinger', 224: 'out-swinger',
    225: 'straight', 236: 'blocked pass', 238: 'fair play', 240: 'gk start',
    241: 'indirect', 266: 'put through', 279: 'kick off', 278: 'tap',
    287: 'over-arm', 307: 'phase of posession ID',
    312: 'phase of possesion start', 23: 'fast break',
    297: 'follows shot rebound', 298: 'follows shot blocked', 15: 'head',
    72: 'left footed', 20: 'right footed', 21: 'other body part',
    22: 'regular play', 23: 'fast break', 24: 'set piece', 25: 'from corner',
    26: 'free kick', 29: 'assisted', 55: 'related event ID',
    96: 'corner situation', 97: 'direct free', 112: 'scramble',
    154: 'intentional assist', 160: 'throw-in set piece',
    216: '2nd related event ID', 233: 'opposite related event ID', 9: 'penalty',
    28: 'own goal', 108: 'volley', 109: 'overhead', 113: 'strong', 114: 'weak',
    115: 'rising', 116: 'dipping', 117: 'lob', 120: 'swerve left',
    121: 'swerve right', 122: 'swerve moving', 133: 'deflection',
    136: 'keeper touched', 137: 'keeper saved', 138: 'hit woodwork',
    153: 'not past goal line', 214: 'big chance', 215: 'individual play',
    217: '2nd assisted', 228: 'own shot blocked', 230: 'gk x coordinate',
    231: 'gk y coordinate', 249: 'temp_shoton', 250: 'temp_blocked',
    251: 'temp_post', 252: 'temp_missed', 253: 'temp_miss not passed goal line',
    254: 'follows a dribble', 261: '1 on 1 clip', 262: 'back heel',
    263: 'direct corner', 280: 'fantasy assist type',
    281: 'fantasy assisted by', 282: 'fantasy assist team', 284: 'duel',
    96: 'corner situation', 110: 'half volley', 111: 'diving header',
    118: 'one bounce', 119: 'few bounces', 316: 'passed penalty',
    16: 'small box-centre', 17: 'box-centre', 18: 'out of box-centre',
    19: '35+ centre', 60: 'small box-right', 61: 'small box-left',
    62: 'box-deep right', 63: 'box-right', 64: 'box-left', 65: 'box-deep left',
    66: 'Out of box-deep right', 67: 'Out of box-right', 68: 'Out of box-left',
    69: 'Out of box-deep', 70: '35+ right', 71: '35+ left', 73: 'Left',
    74: 'High', 75: 'Right', 76: 'Low left', 77: 'High left', 78: 'Low centre',
    79: 'High centre', 80: 'Low right', 81: 'High Right', 82: 'Blocked',
    83: 'Close left', 84: 'Close right', 85: 'Close high', 86: 'Close left and',
    87: 'Close right and', 100: 'Six yard blocke', 101: 'Saved off line',
    102: 'Goal mouth y co', 103: 'Goal mouth z co', 146: 'Blocked x co-or',
    147: 'Blocked y co-or', 276: 'Out on sideline', 300: 'Solo run',
    10: 'Hand', 11: '6-seconds viola', 12: 'Dangerous play', 13: 'Foul',
    31: 'Yellow Card', 32: 'Second yellow', 33: 'Red card', 34: 'Referee abuse',
    35: 'Argument', 36: 'Violent conduct', 37: 'Time wasting',
    38: 'Excessive celeb', 39: 'Crowd interacti', 40: 'Other reason',
    95: 'Back pass', 132: 'Dive', 158: 'Persistent infr',
    159: 'Foul and abusiv', 161: 'Encroachment', 162: 'Leaving field',
    163: 'Entering field', 164: 'Spitting', 165: 'Professional fo',
    166: 'Professional fo', 171: 'Rescinded card', 172: 'No impact on ti',
    184: 'Dissent', 191: 'Off the ball fo', 192: 'Block by hand',
    241: 'Indirect', 242: 'Obstruction', 243: 'Unsporting Beha',
    244: 'Not Retreating', 245: 'Serious Foul', 264: 'Aerial Foul',
    265: 'Attempted Tackl', 289: 'Denied goal-sco', 294: 'Shove/push',
    295: 'Shirt Pull/Hold', 296: 'Elbow/Violent C', 313: 'Illegal Restart',
    314: 'End of offside', 228: 'own shot blocked', 238: 'fair play',
    291: 'other ball contact type', 190: 'From shot off target',
    88 : 'High claim', 89 : '1 on 1', 90 : 'Deflected save',
    91 : 'Dive and deflect', 92 : 'Catch', 93 : 'Dive and catch',
    123: 'Keeper Throw', 124: 'Goal Kick', 128: 'Punch', 139: 'Own Player',
    173: 'Parried safe', 174: 'Parried danger', 175: 'Fingertip', 176: 'Caught',
    177: 'Collected', 178: 'Standing', 179: 'Diving', 180: 'Stooping',
    181: 'Reaching', 182: 'Hands', 183: 'Feet', 198: 'GK hoof',
    199: 'Gk kick from hands', 237: 'Low', 267: 'Right Arm', 268: 'Left Arm',
    269: 'Both Arms', 270: 'Right Leg', 271: 'Left Leg', 272: 'Both Legs',
    273: 'Hit Right Post', 274: 'Hit Left Post', 275: 'Hit Bar',
    232: 'Unchallenged', 301: 'Shot from cross', 186: 'Scored', 187: 'Saved',
    188: 'Missed', 14: 'Last', 94: 'Def', 167: 'Out', 169: 'Leading',
    170: 'Leading', 185: 'Blocked', 239: 'By', 285: 'Defensive',
    286: 'Offensive', 211: 'Overrun', 234: 'Home', 235: 'Away', 288: 'Out',
    30: 'Involved', 41: 'Injury', 42: 'Tactical', 44: 'Player', 59: 'Jersey',
    130: 'Team', 131: 'Team', 145: 'Formation', 194: 'Captain', 197: 'Team',
    283: 'Coach', 290: 'Coach', 292: 'Detailed', 293: 'Position',
    319: 'Captain', 50: 'Official', 51: 'Official', 200: 'Referee',
    201: 'Referee', 208: 'Referee', 247: 'Offside', 248: 'Goal', 53: 'Injured',
    202: 'Weather', 203: 'Crowd', 204: 'Fire', 205: 'Object', 206: 'Spectator',
    207: 'Awaiting', 208: 'Referee', 226: 'Suspended', 227: 'Resume',
    246: 'Drinks', 299: 'Clock', 303: 'Floodlight', 54: 'End', 226: 'Match',
    227: 'Resume', 57: 'end type', 127: 'Direction', 144: 'Explanation',
    189: 'Not', 209: 'Game', 229: 'Post-match', 277: 'Minutes', 302: 'Checks',
    308: 'Goes', 309: 'Goes', 45: 'Temperature', 46: 'Conditions', 47: 'Field',
    48: 'Lightings', 49: 'Attendance', 255: 'Open', 256: 'Air', 257: 'Air',
    258: 'Sold', 259: 'Celsius', 260: 'Floodlight', 56: '56', 285: 'Defensive',
    233: 'Opposite', 41: 'Injury', 304: 'Ball', 305: 'Ball', 306: 'Kit',
    310: 'Player', 311: 'Player', 320: 'Extra'
}