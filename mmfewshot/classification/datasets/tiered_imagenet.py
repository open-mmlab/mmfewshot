import os.path as osp
import pickle
import warnings
from typing import Dict, List, Optional, Sequence, Union

import mmcv
import numpy as np
from mmcls.datasets.builder import DATASETS
from typing_extensions import Literal

from .few_shot_base import FewShotBaseDataset

TRAIN_CLASSES = [
    ('Yorkshire terrier', 'terrier'), ('space shuttle', 'craft'),
    ('drake', 'aquatic bird'),
    ("plane, carpenter's plane, woodworking plane", 'tool'),
    ('mosquito net', 'protective covering, protective cover, protect'),
    ('sax, saxophone', 'musical instrument, instrument'),
    ('container ship, containership, container vessel', 'craft'),
    ('patas, hussar monkey, Erythrocebus patas', 'primate'),
    ('cheetah, chetah, Acinonyx jubatus', 'feline, felid'),
    ('submarine, pigboat, sub, U-boat', 'craft'),
    ('prison, prison house', 'establishment'),
    ('can opener, tin opener', 'tool'), ('syringe', 'instrument'),
    ('odometer, hodometer, mileometer, milometer', 'instrument'),
    ('bassoon', 'musical instrument, instrument'),
    ('Kerry blue terrier', 'terrier'),
    ('scale, weighing machine', 'instrument'), ('baseball', 'game equipment'),
    ('cassette player', 'electronic equipment'),
    ('shield, buckler', 'protective covering, protective cover, protect'),
    ('goldfinch, Carduelis carduelis', 'passerine, passeriform bird'),
    ('cornet, horn, trumpet, trump', 'musical instrument, instrument'),
    ('flute, transverse flute', 'musical instrument, instrument'),
    ('stopwatch, stop watch', 'instrument'), ('basketball', 'game equipment'),
    ('brassiere, bra, bandeau', 'garment'),
    ('bulbul', 'passerine, passeriform bird'),
    ('steel drum', 'musical instrument, instrument'),
    ('bolo tie, bolo, bola tie, bola', 'garment'),
    ('planetarium', 'building, edifice'), ('stethoscope', 'instrument'),
    ('proboscis monkey, Nasalis larvatus', 'primate'),
    ('guillotine', 'instrument'),
    ('Scottish deerhound, deerhound', 'hound, hound dog'),
    ('ocarina, sweet potato', 'musical instrument, instrument'),
    ('Border terrier', 'terrier'),
    ('capuchin, ringtail, Cebus capucinus', 'primate'),
    ('magnetic compass', 'instrument'), ('alligator lizard', 'saurian'),
    ('baboon', 'primate'), ('sundial', 'instrument'),
    ('gibbon, Hylobates lar', 'primate'),
    ('grand piano, grand', 'musical instrument, instrument'),
    ('Arabian camel, dromedary, Camelus dromedarius',
     'ungulate, hoofed mammal'), ('basset, basset hound', 'hound, hound dog'),
    ('corkscrew, bottle screw', 'tool'), ('miniskirt, mini', 'garment'),
    ('missile', 'instrument'), ('hatchet', 'tool'),
    ('acoustic guitar', 'musical instrument, instrument'),
    ('impala, Aepyceros melampus', 'ungulate, hoofed mammal'),
    ('parking meter', 'instrument'),
    ('greenhouse, nursery, glasshouse', 'building, edifice'),
    ('home theater, home theatre', 'building, edifice'),
    ('hartebeest', 'ungulate, hoofed mammal'),
    ('hippopotamus, hippo, river horse, Hippopotamus amphibius',
     'ungulate, hoofed mammal'), ('warplane, military plane', 'craft'),
    ('albatross, mollymawk', 'aquatic bird'),
    ('umbrella', 'protective covering, protective cover, protect'),
    ('shoe shop, shoe-shop, shoe store', 'establishment'),
    ('suit, suit of clothes', 'garment'),
    ('pickelhaube', 'protective covering, protective cover, protect'),
    ('soccer ball', 'game equipment'), ('yawl', 'craft'),
    ('screwdriver', 'tool'),
    ('Madagascar cat, ring-tailed lemur, Lemur catta', 'primate'),
    ('garter snake, grass snake', 'snake, serpent, ophidian'),
    ('bustard', 'aquatic bird'), ('tabby, tabby cat', 'feline, felid'),
    ('airliner', 'craft'),
    ('tobacco shop, tobacconist shop, tobacconist', 'establishment'),
    ('Italian greyhound', 'hound, hound dog'), ('projector', 'instrument'),
    ('bittern', 'aquatic bird'), ('rifle', 'instrument'),
    ('pay-phone, pay-station', 'electronic equipment'),
    ('house finch, linnet, Carpodacus mexicanus',
     'passerine, passeriform bird'), ('monastery', 'building, edifice'),
    ('lens cap, lens cover', 'protective covering, protective cover, protect'),
    ('maillot, tank suit', 'garment'), ('canoe', 'craft'),
    ('letter opener, paper knife, paperknife', 'tool'),
    ('nail', 'restraint, constraint'), ('guenon, guenon monkey', 'primate'),
    ('CD player', 'electronic equipment'),
    ('safety pin', 'restraint, constraint'),
    ('harp', 'musical instrument, instrument'),
    ('disk brake, disc brake', 'restraint, constraint'),
    ('otterhound, otter hound', 'hound, hound dog'),
    ('green mamba', 'snake, serpent, ophidian'),
    ('violin, fiddle', 'musical instrument, instrument'),
    ('American coot, marsh hen, mud hen, water hen, Fulica americana',
     'aquatic bird'), ('ram, tup', 'ungulate, hoofed mammal'),
    ('jay', 'passerine, passeriform bird'), ('trench coat', 'garment'),
    ('Indian cobra, Naja naja', 'snake, serpent, ophidian'),
    ('projectile, missile', 'instrument'), ('schooner', 'craft'),
    ('magpie', 'passerine, passeriform bird'), ('Norwich terrier', 'terrier'),
    ('cairn, cairn terrier', 'terrier'),
    ('crossword puzzle, crossword', 'game equipment'),
    ('snow leopard, ounce, Panthera uncia', 'feline, felid'),
    ('gong, tam-tam', 'musical instrument, instrument'),
    ('library', 'building, edifice'),
    ('swimming trunks, bathing trunks', 'garment'),
    ('Staffordshire bullterrier, Staffordshire bull terrier', 'terrier'),
    ('Lakeland terrier', 'terrier'),
    ('black stork, Ciconia nigra', 'aquatic bird'),
    ('king penguin, Aptenodytes patagonica', 'aquatic bird'),
    ('water ouzel, dipper', 'passerine, passeriform bird'),
    ('macaque', 'primate'), ('lynx, catamount', 'feline, felid'),
    ('ping-pong ball', 'game equipment'), ('standard schnauzer', 'terrier'),
    ('Australian terrier', 'terrier'), ('stupa, tope', 'building, edifice'),
    ('white stork, Ciconia ciconia', 'aquatic bird'),
    ('king snake, kingsnake', 'snake, serpent, ophidian'),
    ('Airedale, Airedale terrier', 'terrier'),
    ('banjo', 'musical instrument, instrument'), ('Windsor tie', 'garment'),
    ('abaya', 'garment'), ('stole', 'garment'),
    ('vine snake', 'snake, serpent, ophidian'),
    ('Bedlington terrier', 'terrier'), ('langur', 'primate'),
    ('catamaran', 'craft'), ('sarong', 'garment'),
    ('spoonbill', 'aquatic bird'),
    ('boa constrictor, Constrictor constrictor', 'snake, serpent, ophidian'),
    ('ruddy turnstone, Arenaria interpres', 'aquatic bird'),
    ('hognose snake, puff adder, sand viper', 'snake, serpent, ophidian'),
    ('American chameleon, anole, Anolis carolinensis', 'saurian'),
    ('rugby ball', 'game equipment'),
    ('black swan, Cygnus atratus', 'aquatic bird'),
    ('frilled lizard, Chlamydosaurus kingi', 'saurian'),
    ('oscilloscope, scope, cathode-ray oscilloscope, CRO',
     'electronic equipment'),
    ('ski mask', 'protective covering, protective cover, protect'),
    ('marmoset', 'primate'),
    ('Komodo dragon, Komodo lizard, dragon lizard, giant lizard, '
     'Varanus komodoensis', 'saurian'),
    ('accordion, piano accordion, squeeze box',
     'musical instrument, instrument'),
    ('horned viper, cerastes, sand viper, horned asp, Cerastes cornutus',
     'snake, serpent, ophidian'),
    ('bookshop, bookstore, bookstall', 'establishment'),
    ('Boston bull, Boston terrier', 'terrier'), ('crane', 'aquatic bird'),
    ('junco, snowbird', 'passerine, passeriform bird'),
    ('silky terrier, Sydney silky', 'terrier'),
    ('Egyptian cat', 'feline, felid'), ('Irish terrier', 'terrier'),
    ('leopard, Panthera pardus', 'feline, felid'),
    ('sea snake', 'snake, serpent, ophidian'),
    ('hog, pig, grunter, squealer, Sus scrofa', 'ungulate, hoofed mammal'),
    ('colobus, colobus monkey', 'primate'),
    ('chickadee', 'passerine, passeriform bird'),
    ('Scotch terrier, Scottish terrier, Scottie', 'terrier'),
    ('digital watch', 'instrument'), ('analog clock', 'instrument'),
    ('zebra', 'ungulate, hoofed mammal'),
    ('American Staffordshire terrier, Staffordshire terrier, '
     'American pit bull terrier, pit bull terrier', 'terrier'),
    ('European gallinule, Porphyrio porphyrio', 'aquatic bird'),
    ('lampshade, lamp shade',
     'protective covering, protective cover, protect'),
    ('holster', 'protective covering, protective cover, protect'),
    ('jaguar, panther, Panthera onca, Felis onca', 'feline, felid'),
    ('cleaver, meat cleaver, chopper', 'tool'),
    ('brambling, Fringilla montifringilla', 'passerine, passeriform bird'),
    ('orangutan, orang, orangutang, Pongo pygmaeus', 'primate'),
    ('combination lock', 'restraint, constraint'),
    ('tile roof', 'protective covering, protective cover, protect'),
    ('borzoi, Russian wolfhound', 'hound, hound dog'),
    ('water snake', 'snake, serpent, ophidian'),
    ('knot', 'restraint, constraint'),
    ('window shade', 'protective covering, protective cover, protect'),
    ('mosque', 'building, edifice'),
    ('Walker hound, Walker foxhound', 'hound, hound dog'),
    ('cardigan', 'garment'), ('warthog', 'ungulate, hoofed mammal'),
    ('whiptail, whiptail lizard', 'saurian'), ('plow, plough', 'tool'),
    ('bluetick', 'hound, hound dog'), ('poncho', 'garment'),
    ('shovel', 'tool'),
    ('sidewinder, horned rattlesnake, Crotalus cerastes',
     'snake, serpent, ophidian'), ('croquet ball', 'game equipment'),
    ('sorrel', 'ungulate, hoofed mammal'), ('airship, dirigible', 'craft'),
    ('goose', 'aquatic bird'), ('church, church building',
                                'building, edifice'),
    ('titi, titi monkey', 'primate'),
    ('butcher shop, meat market', 'establishment'),
    ('diamondback, diamondback rattlesnake, Crotalus adamanteus',
     'snake, serpent, ophidian'),
    ('common iguana, iguana, Iguana iguana', 'saurian'),
    ('Saluki, gazelle hound', 'hound, hound dog'),
    ('monitor', 'electronic equipment'),
    ('sunglasses, dark glasses, shades', 'instrument'),
    ('flamingo', 'aquatic bird'),
    ('seat belt, seatbelt', 'restraint, constraint'),
    ('Persian cat', 'feline, felid'), ('gorilla, Gorilla gorilla', 'primate'),
    ('banded gecko', 'saurian'),
    ('thatch, thatched roof',
     'protective covering, protective cover, protect'),
    ('beagle', 'hound, hound dog'), ('limpkin, Aramus pictus', 'aquatic bird'),
    ('jigsaw puzzle', 'game equipment'), ('rule, ruler', 'instrument'),
    ('hammer', 'tool'), ('cello, violoncello',
                         'musical instrument, instrument'),
    ('lab coat, laboratory coat', 'garment'),
    ('indri, indris, Indri indri, Indri brevicaudatus', 'primate'),
    ('vault', 'protective covering, protective cover, protect'),
    ('cellular telephone, cellular phone, cellphone, cell, mobile phone',
     'electronic equipment'), ('whippet', 'hound, hound dog'),
    ('siamang, Hylobates syndactylus, Symphalangus syndactylus', 'primate'),
    ("loupe, jeweler's loupe", 'instrument'), ('modem',
                                               'electronic equipment'),
    ('lifeboat', 'craft'),
    ('dial telephone, dial phone', 'electronic equipment'),
    ('cougar, puma, catamount, mountain lion, painter, panther, '
     'Felis concolor', 'feline, felid'),
    ('thimble', 'protective covering, protective cover, protect'),
    ('ibex, Capra ibex', 'ungulate, hoofed mammal'),
    ('lawn mower, mower', 'tool'),
    ('bell cote, bell cot', 'protective covering, protective cover, protect'),
    ('chain mail, ring mail, mail, chain armor, chain armour, ring armor, '
     'ring armour', 'protective covering, protective cover, protect'),
    ('hair slide', 'restraint, constraint'),
    ('apiary, bee house', 'building, edifice'),
    ('harmonica, mouth organ, harp, mouth harp',
     'musical instrument, instrument'),
    ('green snake, grass snake', 'snake, serpent, ophidian'),
    ('howler monkey, howler', 'primate'), ('digital clock', 'instrument'),
    ('restaurant, eating house, eating place, eatery', 'building, edifice'),
    ('miniature schnauzer', 'terrier'),
    ('panpipe, pandean pipe, syrinx', 'musical instrument, instrument'),
    ('pirate, pirate ship', 'craft'),
    ('window screen', 'protective covering, protective cover, protect'),
    ('binoculars, field glasses, opera glasses', 'instrument'),
    ('Afghan hound, Afghan', 'hound, hound dog'),
    ('cinema, movie theater, movie theatre, movie house, picture palace',
     'building, edifice'), ('liner, ocean liner', 'craft'),
    ('ringneck snake, ring-necked snake, ring snake',
     'snake, serpent, ophidian'), ('redshank, Tringa totanus', 'aquatic bird'),
    ('Siamese cat, Siamese', 'feline, felid'),
    ('thunder snake, worm snake, Carphophis amoenus',
     'snake, serpent, ophidian'), ('boathouse', 'building, edifice'),
    ('jersey, T-shirt, tee shirt', 'garment'),
    ('soft-coated wheaten terrier', 'terrier'),
    ('scabbard', 'protective covering, protective cover, protect'),
    ('muzzle', 'restraint, constraint'),
    ('Ibizan hound, Ibizan Podenco', 'hound, hound dog'),
    ('tennis ball', 'game equipment'), ('padlock', 'restraint, constraint'),
    ('kimono', 'garment'), ('redbone', 'hound, hound dog'),
    ('wild boar, boar, Sus scrofa', 'ungulate, hoofed mammal'),
    ('dowitcher', 'aquatic bird'),
    ('oboe, hautboy, hautbois', 'musical instrument, instrument'),
    ('electric guitar', 'musical instrument, instrument'), ('trimaran',
                                                            'craft'),
    ('barometer', 'instrument'), ('llama', 'ungulate, hoofed mammal'),
    ('robin, American robin, Turdus migratorius',
     'passerine, passeriform bird'),
    ('maraca', 'musical instrument, instrument'),
    ('feather boa, boa', 'garment'),
    ('Dandie Dinmont, Dandie Dinmont terrier', 'terrier'),
    ('Lhasa, Lhasa apso', 'terrier'), ('bow', 'instrument'),
    ('punching bag, punch bag, punching ball, punchball', 'game equipment'),
    ('volleyball', 'game equipment'), ('Norfolk terrier', 'terrier'),
    ('Gila monster, Heloderma suspectum', 'saurian'),
    ('fire screen, fireguard',
     'protective covering, protective cover, protect'),
    ('hourglass', 'instrument'),
    ('chimpanzee, chimp, Pan troglodytes', 'primate'),
    ('birdhouse', 'protective covering, protective cover, protect'),
    ('Sealyham terrier, Sealyham', 'terrier'),
    ('Tibetan terrier, chrysanthemum dog', 'terrier'),
    ('palace', 'building, edifice'), ('wreck', 'craft'),
    ('overskirt', 'garment'), ('pelican', 'aquatic bird'),
    ('French horn, horn', 'musical instrument, instrument'),
    ('tiger cat', 'feline, felid'), ('barbershop', 'establishment'),
    ('revolver, six-gun, six-shooter', 'instrument'),
    ('Irish wolfhound', 'hound, hound dog'),
    ('lion, king of beasts, Panthera leo', 'feline, felid'),
    ('fur coat', 'garment'), ('ox', 'ungulate, hoofed mammal'),
    ('cuirass', 'protective covering, protective cover, protect'),
    ('grocery store, grocery, food market, market', 'establishment'),
    ('hoopskirt, crinoline', 'garment'),
    ('spider monkey, Ateles geoffroyi', 'primate'),
    ('tiger, Panthera tigris', 'feline, felid'),
    ('bloodhound, sleuthhound', 'hound, hound dog'),
    ('red-backed sandpiper, dunlin, Erolia alpina', 'aquatic bird'),
    ('drum, membranophone, tympan', 'musical instrument, instrument'),
    ('radio telescope, radio reflector', 'instrument'),
    ('West Highland white terrier', 'terrier'),
    ('bow tie, bow-tie, bowtie', 'garment'), ('golf ball', 'game equipment'),
    ('barn', 'building, edifice'),
    ('binder, ring-binder', 'protective covering, protective cover, protect'),
    ('English foxhound', 'hound, hound dog'),
    ('bison', 'ungulate, hoofed mammal'), ('screw', 'restraint, constraint'),
    ('assault rifle, assault gun', 'instrument'),
    ('diaper, nappy, napkin', 'garment'),
    ('bighorn, bighorn sheep, cimarron, Rocky Mountain bighorn, '
     'Rocky Mountain sheep, Ovis canadensis', 'ungulate, hoofed mammal'),
    ('Weimaraner', 'hound, hound dog'),
    ('computer keyboard, keypad', 'electronic equipment'),
    ('black-and-tan coonhound', 'hound, hound dog'),
    ('little blue heron, Egretta caerulea', 'aquatic bird'),
    ('breastplate, aegis, egis',
     'protective covering, protective cover, protect'),
    ('gasmask, respirator, gas helmet',
     'protective covering, protective cover, protect'),
    ('aircraft carrier, carrier, flattop, attack aircraft carrier', 'craft'),
    ('iPod', 'electronic equipment'),
    ('organ, pipe organ', 'musical instrument, instrument'),
    ('wall clock', 'instrument'),
    ('rock python, rock snake, Python sebae', 'snake, serpent, ophidian'),
    ('squirrel monkey, Saimiri sciureus', 'primate'),
    ('bikini, two-piece', 'garment'),
    ('water buffalo, water ox, Asiatic buffalo, Bubalus bubalis',
     'ungulate, hoofed mammal'),
    ('upright, upright piano', 'musical instrument, instrument'),
    ('chime, bell, gong', 'musical instrument, instrument'),
    ('confectionery, confectionary, candy store', 'establishment'),
    ('indigo bunting, indigo finch, indigo bird, Passerina cyanea',
     'passerine, passeriform bird'),
    ('green lizard, Lacerta viridis', 'saurian'),
    ('Norwegian elkhound, elkhound', 'hound, hound dog'),
    ('dome', 'protective covering, protective cover, protect'),
    ('buckle', 'restraint, constraint'), ('giant schnauzer', 'terrier'),
    ('jean, blue jean, denim', 'garment'),
    ('wire-haired fox terrier', 'terrier'),
    ('African chameleon, Chamaeleo chamaeleon', 'saurian'),
    ('trombone', 'musical instrument, instrument'),
    ('oystercatcher, oyster catcher', 'aquatic bird'), ('sweatshirt',
                                                        'garment'),
    ('American egret, great white heron, Egretta albus', 'aquatic bird'),
    ('marimba, xylophone', 'musical instrument, instrument'),
    ('gazelle', 'ungulate, hoofed mammal'),
    ('red-breasted merganser, Mergus serrator', 'aquatic bird'),
    ('tape player', 'electronic equipment'), ('speedboat', 'craft'),
    ('gondola', 'craft'),
    ('night snake, Hypsiglena torquata', 'snake, serpent, ophidian'),
    ('cannon', 'instrument'), ("plunger, plumber's helper", 'tool'),
    ('balloon', 'craft'), ('toyshop', 'establishment'), ('agama', 'saurian'),
    ('fireboat', 'craft'), ('bakery, bakeshop, bakehouse', 'establishment')
]
VAL_CLASSES = [
    ('cab, hack, taxi, taxicab', 'motor vehicle, automotive vehicle'),
    ('jeep, landrover', 'motor vehicle, automotive vehicle'),
    ('English setter', 'sporting dog, gun dog'),
    ('flat-coated retriever', 'sporting dog, gun dog'),
    ('bassinet', 'furnishing'),
    ('sports car, sport car', 'motor vehicle, automotive vehicle'),
    ('golfcart, golf cart', 'motor vehicle, automotive vehicle'),
    ('clumber, clumber spaniel', 'sporting dog, gun dog'),
    ('puck, hockey puck', 'mechanism'), ('reel', 'mechanism'),
    ('Welsh springer spaniel', 'sporting dog, gun dog'),
    ('car wheel', 'mechanism'), ('wardrobe, closet, press', 'furnishing'),
    ('go-kart', 'motor vehicle, automotive vehicle'),
    ('switch, electric switch, electrical switch', 'mechanism'),
    ('crib, cot', 'furnishing'), ('laptop, laptop computer', 'machine'),
    ('thresher, thrasher, threshing machine', 'machine'),
    ('web site, website, internet site, site', 'machine'),
    ('English springer, English springer spaniel', 'sporting dog, gun dog'),
    ('iron, smoothing iron', 'durables, durable goods, consumer durables'),
    ('Gordon setter', 'sporting dog, gun dog'),
    ('Labrador retriever', 'sporting dog, gun dog'),
    ('Irish water spaniel', 'sporting dog, gun dog'),
    ('amphibian, amphibious vehicle', 'motor vehicle, automotive vehicle'),
    ('file, file cabinet, filing cabinet', 'furnishing'),
    ('harvester, reaper', 'machine'),
    ('convertible', 'motor vehicle, automotive vehicle'),
    ('paddlewheel, paddle wheel', 'mechanism'),
    ('microwave, microwave oven',
     'durables, durable goods, consumer durables'), ('swing', 'mechanism'),
    ('chiffonier, commode', 'furnishing'), ('desktop computer', 'machine'),
    ('gas pump, gasoline pump, petrol pump, island dispenser', 'mechanism'),
    ('beach wagon, station wagon, wagon, estate car, beach waggon, station '
     'waggon, waggon', 'motor vehicle, automotive vehicle'),
    ('carousel, carrousel, merry-go-round, roundabout, whirligig',
     'mechanism'), ("potter's wheel", 'mechanism'),
    ('folding chair', 'furnishing'),
    ('fire engine, fire truck', 'motor vehicle, automotive vehicle'),
    ('slide rule, slipstick', 'machine'),
    ('vizsla, Hungarian pointer', 'sporting dog, gun dog'),
    ('waffle iron', 'durables, durable goods, consumer durables'),
    ('trailer truck, tractor trailer, trucking rig, rig, articulated lorry, '
     'semi', 'motor vehicle, automotive vehicle'),
    ('toilet seat', 'furnishing'),
    ('medicine chest, medicine cabinet', 'furnishing'),
    ('Brittany spaniel', 'sporting dog, gun dog'),
    ('Chesapeake Bay retriever', 'sporting dog, gun dog'),
    ('cash machine, cash dispenser, automated teller machine, automatic '
     'teller machine, automated teller, automatic teller, ATM', 'machine'),
    ('moped', 'motor vehicle, automotive vehicle'),
    ('Model T', 'motor vehicle, automotive vehicle'),
    ('bookcase', 'furnishing'),
    ('ambulance', 'motor vehicle, automotive vehicle'),
    ('German short-haired pointer', 'sporting dog, gun dog'),
    ('dining table, board', 'furnishing'),
    ('minivan', 'motor vehicle, automotive vehicle'),
    ('police van, police wagon, paddy wagon, patrol wagon, wagon, '
     'black Maria', 'motor vehicle, automotive vehicle'),
    ('entertainment center', 'furnishing'), ('throne', 'furnishing'),
    ('desk', 'furnishing'), ('notebook, notebook computer', 'machine'),
    ('snowplow, snowplough', 'motor vehicle, automotive vehicle'),
    ('cradle', 'furnishing'), ('abacus', 'machine'),
    ('hand-held computer, hand-held microcomputer', 'machine'),
    ('Dutch oven', 'durables, durable goods, consumer durables'),
    ('toaster', 'durables, durable goods, consumer durables'),
    ('barber chair', 'furnishing'), ('vending machine', 'machine'),
    ('four-poster', 'furnishing'),
    ('rotisserie', 'durables, durable goods, consumer durables'),
    ('hook, claw', 'mechanism'),
    ('vacuum, vacuum cleaner', 'durables, durable goods, consumer durables'),
    ('pickup, pickup truck', 'motor vehicle, automotive vehicle'),
    ('table lamp', 'furnishing'), ('rocking chair, rocker', 'furnishing'),
    ('prayer rug, prayer mat', 'furnishing'),
    ('moving van', 'motor vehicle, automotive vehicle'),
    ('studio couch, day bed', 'furnishing'),
    ('racer, race car, racing car', 'motor vehicle, automotive vehicle'),
    ('park bench', 'furnishing'),
    ('Irish setter, red setter', 'sporting dog, gun dog'),
    ('refrigerator, icebox', 'durables, durable goods, consumer durables'),
    ('china cabinet, china closet', 'furnishing'),
    ('cocker spaniel, English cocker spaniel, cocker',
     'sporting dog, gun dog'), ('radiator', 'mechanism'),
    ('Sussex spaniel', 'sporting dog, gun dog'),
    ('hand blower, blow dryer, blow drier, hair dryer, hair drier',
     'durables, durable goods, consumer durables'),
    ('slot, one-armed bandit', 'machine'),
    ('golden retriever', 'sporting dog, gun dog'),
    ('curly-coated retriever', 'sporting dog, gun dog'),
    ('limousine, limo', 'motor vehicle, automotive vehicle'),
    ('washer, automatic washer, washing machine',
     'durables, durable goods, consumer durables'),
    ('garbage truck, dustcart', 'motor vehicle, automotive vehicle'),
    ('dishwasher, dish washer, dishwashing machine',
     'durables, durable goods, consumer durables'), ('pinwheel', 'mechanism'),
    ('espresso maker', 'durables, durable goods, consumer durables'),
    ('tow truck, tow car, wrecker', 'motor vehicle, automotive vehicle')
]
TEST_CLASSES = [
    ('Siberian husky', 'working dog'), ('dung beetle', 'insect'),
    ('jackfruit, jak, jack', 'solid'), ('miniature pinscher', 'working dog'),
    ('tiger shark, Galeocerdo cuvieri', 'aquatic vertebrate'),
    ('weevil', 'insect'),
    ('goldfish, Carassius auratus', 'aquatic vertebrate'),
    ('schipperke', 'working dog'), ('Tibetan mastiff', 'working dog'),
    ('orange', 'solid'), ('whiskey jug', 'vessel'),
    ('hammerhead, hammerhead shark', 'aquatic vertebrate'),
    ('bull mastiff', 'working dog'), ('eggnog', 'substance'),
    ('bee', 'insect'), ('tench, Tinca tinca', 'aquatic vertebrate'),
    ('chocolate sauce, chocolate syrup', 'substance'),
    ("dragonfly, darning needle, devil's darning needle, sewing needle, "
     'snake feeder, snake doctor, mosquito hawk, skeeter hawk', 'insect'),
    ('zucchini, courgette', 'solid'), ('kelpie', 'working dog'),
    ('stone wall', 'obstruction, obstructor, obstructer, impedimen'),
    ('butternut squash', 'solid'), ('mushroom', 'solid'),
    ('Old English sheepdog, bobtail', 'working dog'),
    ('dam, dike, dyke', 'obstruction, obstructor, obstructer, impedimen'),
    ('picket fence, paling', 'obstruction, obstructor, obstructer, impedimen'),
    ('espresso', 'substance'), ('beer bottle', 'vessel'),
    ('plate', 'substance'), ('dough', 'substance'),
    ('sandbar, sand bar', 'geological formation, formation'),
    ('boxer', 'working dog'), ('bathtub, bathing tub, bath, tub', 'vessel'),
    ('beaker', 'vessel'), ('bucket, pail', 'vessel'),
    ('Border collie', 'working dog'), ('sturgeon', 'aquatic vertebrate'),
    ('worm fence, snake fence, snake-rail fence, Virginia fence',
     'obstruction, obstructor, obstructer, impedimen'),
    ('seashore, coast, seacoast, sea-coast',
     'geological formation, formation'),
    ('long-horned beetle, longicorn, longicorn beetle', 'insect'),
    ('turnstile', 'obstruction, obstructor, obstructer, impedimen'),
    ('groenendael', 'working dog'), ('vase', 'vessel'), ('teapot', 'vessel'),
    ('water tower', 'vessel'), ('strawberry', 'solid'), ('burrito',
                                                         'substance'),
    ('cauliflower', 'solid'), ('volcano', 'geological formation, formation'),
    ('valley, vale', 'geological formation, formation'),
    ('head cabbage', 'solid'), ('tub, vat', 'vessel'),
    ('lacewing, lacewing fly', 'insect'),
    ('coral reef', 'geological formation, formation'),
    ('hot pot, hotpot', 'substance'), ('custard apple', 'solid'),
    ('monarch, monarch butterfly, milkweed butterfly, Danaus plexippus',
     'insect'), ('cricket', 'insect'), ('pill bottle', 'vessel'),
    ('walking stick, walkingstick, stick insect', 'insect'),
    ('promontory, headland, head, foreland',
     'geological formation, formation'), ('malinois', 'working dog'),
    ('pizza, pizza pie', 'substance'),
    ('malamute, malemute, Alaskan malamute', 'working dog'),
    ('kuvasz', 'working dog'), ('trifle', 'substance'), ('fig', 'solid'),
    ('komondor', 'working dog'), ('ant, emmet, pismire', 'insect'),
    ('electric ray, crampfish, numbfish, torpedo', 'aquatic vertebrate'),
    ('Granny Smith', 'solid'), ('cockroach, roach', 'insect'),
    ('stingray', 'aquatic vertebrate'), ('red wine', 'substance'),
    ('Saint Bernard, St Bernard', 'working dog'),
    ('ice lolly, lolly, lollipop, popsicle', 'substance'),
    ('bell pepper', 'solid'), ('cup', 'substance'), ('pomegranate', 'solid'),
    ('Appenzeller', 'working dog'), ('hay', 'substance'),
    ('EntleBucher', 'working dog'),
    ('sulphur butterfly, sulfur butterfly', 'insect'),
    ('mantis, mantid', 'insect'), ('Bernese mountain dog', 'working dog'),
    ('banana', 'solid'), ('water jug', 'vessel'), ('cicada, cicala', 'insect'),
    ('barracouta, snoek', 'aquatic vertebrate'),
    ('washbasin, handbasin, washbowl, lavabo, wash-hand basin', 'vessel'),
    ('wine bottle', 'vessel'), ('Rottweiler', 'working dog'),
    ('briard', 'working dog'),
    ('puffer, pufferfish, blowfish, globefish', 'aquatic vertebrate'),
    ('ground beetle, carabid beetle', 'insect'),
    ('Bouvier des Flandres, Bouviers des Flandres', 'working dog'),
    ('chainlink fence', 'obstruction, obstructor, obstructer, impedimen'),
    ('damselfly', 'insect'), ('grasshopper, hopper', 'insect'),
    ('carbonara', 'substance'),
    ('German shepherd, German shepherd dog, German police dog, alsatian',
     'working dog'), ('guacamole', 'substance'),
    ('leaf beetle, chrysomelid', 'insect'), ('caldron, cauldron', 'vessel'),
    ('fly', 'insect'),
    ('bannister, banister, balustrade, balusters, handrail',
     'obstruction, obstructor, obstructer, impedimen'),
    ('spaghetti squash', 'solid'), ('coffee mug', 'vessel'),
    ('gar, garfish, garpike, billfish, Lepisosteus osseus',
     'aquatic vertebrate'), ('barrel, cask', 'vessel'),
    ('eel', 'aquatic vertebrate'), ('rain barrel', 'vessel'),
    ('coho, cohoe, coho salmon, blue jack, silver salmon, '
     'Oncorhynchus kisutch', 'aquatic vertebrate'), ('water bottle', 'vessel'),
    ('menu', 'substance'), ('tiger beetle', 'insect'),
    ('Great Dane', 'working dog'),
    ('rock beauty, Holocanthus tricolor', 'aquatic vertebrate'),
    ('anemone fish', 'aquatic vertebrate'), ('mortar', 'vessel'),
    ('Eskimo dog, husky', 'working dog'),
    ('affenpinscher, monkey pinscher, monkey dog', 'working dog'),
    ('breakwater, groin, groyne, mole, bulwark, seawall, jetty',
     'obstruction, obstructor, obstructer, impedimen'),
    ('artichoke, globe artichoke', 'solid'), ('broccoli', 'solid'),
    ('French bulldog', 'working dog'), ('coffeepot', 'vessel'),
    ('cliff, drop, drop-off', 'geological formation, formation'),
    ('ladle', 'vessel'),
    ('sliding door', 'obstruction, obstructor, obstructer, impedimen'),
    ('leafhopper', 'insect'), ('collie', 'working dog'),
    ('Doberman, Doberman pinscher', 'working dog'), ('pitcher, ewer',
                                                     'vessel'),
    ('admiral', 'insect'), ('cabbage butterfly', 'insect'),
    ('geyser', 'geological formation, formation'), ('cheeseburger',
                                                    'substance'),
    ('grille, radiator grille',
     'obstruction, obstructor, obstructer, impedimen'),
    ('ladybug, ladybeetle, lady beetle, ladybird, ladybird beetle', 'insect'),
    ('great white shark, white shark, man-eater, man-eating shark, '
     'Carcharodon carcharias', 'aquatic vertebrate'),
    ('pineapple, ananas', 'solid'), ('cardoon', 'solid'),
    ('pop bottle, soda bottle', 'vessel'), ('lionfish', 'aquatic vertebrate'),
    ('cucumber, cuke', 'solid'), ('face powder', 'substance'),
    ('Shetland sheepdog, Shetland sheep dog, Shetland', 'working dog'),
    ('ringlet, ringlet butterfly', 'insect'),
    ('Greater Swiss Mountain dog', 'working dog'),
    ('alp', 'geological formation, formation'), ('consomme', 'substance'),
    ('potpie', 'substance'), ('acorn squash', 'solid'),
    ('ice cream, icecream', 'substance'),
    ('lakeside, lakeshore', 'geological formation, formation'),
    ('hotdog, hot dog, red hot', 'substance'), ('rhinoceros beetle', 'insect'),
    ('lycaenid, lycaenid butterfly', 'insect'), ('lemon', 'solid')
]


@DATASETS.register_module()
class TieredImageNetDataset(FewShotBaseDataset):
    """TieredImageNet dataset for few shot classification.

    Args:
        subset (str| list[str]): The classes of whole dataset are split into
            three disjoint subset: train, val and test. If subset is a string,
            only one subset data will be loaded. If subset is a list of
            string, then all data of subset in list will be loaded.
            Options: ['train', 'val', 'test']. Default: 'train'.
    """

    resource = 'https://github.com/renmengye/few-shot-ssl-public'
    TRAIN_CLASSES = TRAIN_CLASSES
    VAL_CLASSES = VAL_CLASSES
    TEST_CLASSES = TEST_CLASSES

    def __init__(self,
                 subset: Literal['train', 'test', 'val'] = 'train',
                 *args,
                 **kwargs):
        if isinstance(subset, str):
            subset = [subset]
        for subset_ in subset:
            assert subset_ in ['train', 'test', 'val']
        self.subset = subset
        self.GENERAL_CLASSES = self.get_general_classes()
        super().__init__(*args, **kwargs)

    def get_classes(
            self,
            classes: Optional[Union[Sequence[str],
                                    str]] = None) -> Sequence[str]:
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): Three types of input
            will correspond to different processing logics:

            - If `classes` is a tuple or list, it will override the CLASSES
              predefined in the dataset.
            - If `classes` is None, we directly use pre-defined CLASSES will
              be used by the dataset.
            - If `classes` is a string, it is the path of a classes file that
              contains the name of all classes. Each line of the file contains
              a single class name.

        Returns:
            tuple[str] or list[str]: Names of categories of the dataset.
        """
        if classes is None:
            class_names = []
            for subset_ in self.subset:
                if subset_ == 'train':
                    class_names += [i[0] for i in self.TRAIN_CLASSES]
                elif subset_ == 'val':
                    class_names += [i[0] for i in self.VAL_CLASSES]
                elif subset_ == 'test':
                    class_names += [i[0] for i in self.TEST_CLASSES]
                else:
                    raise ValueError(f'invalid subset {subset_} only '
                                     f'support train, val or test.')
        elif isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')
        return class_names

    def get_general_classes(self) -> List[str]:
        """Get general classes of each classes."""
        general_classes = []
        for subset_ in self.subset:
            if subset_ == 'train':
                general_classes += [i[1] for i in self.TRAIN_CLASSES]
            elif subset_ == 'val':
                general_classes += [i[1] for i in self.VAL_CLASSES]
            elif subset_ == 'test':
                general_classes += [i[1] for i in self.TEST_CLASSES]
            else:
                raise ValueError(f'invalid subset {subset_} only '
                                 f'support train, val or test.')
        return general_classes

    def load_annotations(self) -> List[Dict]:
        """Load annotation according to the classes subset."""
        data_infos = []
        for subset_ in self.subset:
            labels_file = osp.join(self.data_prefix, f'{subset_}_labels.pkl')
            img_bytes_file = osp.join(self.data_prefix,
                                      f'{subset_}_images_png.pkl')
            assert osp.exists(img_bytes_file) and osp.exists(labels_file), \
                f'Please download ann_file through {self.resource}.'
            data_infos = []
            with open(labels_file, 'rb') as labels, \
                    open(img_bytes_file, 'rb') as img_bytes:
                labels = pickle.load(labels)
                img_bytes = pickle.load(img_bytes)
                label_specific = labels['label_specific']
                label_general = labels['label_general']
                class_specific = labels['label_specific_str']
                class_general = labels['label_general_str']
                unzip_file_path = osp.join(self.data_prefix, subset_)
                is_unzip_file = osp.exists(unzip_file_path)
                if not is_unzip_file:
                    msg = 'Please unzip pickle file first by provided ' \
                          'script in tools. Otherwise the whole pickle ' \
                          'file may cost heavy memory usage.'
                    warnings.warn(msg)
                for i in range(len(img_bytes)):
                    class_specific_name = class_specific[label_specific[i]]
                    class_general_name = class_general[label_general[i]]
                    gt_label = self.class_to_idx[class_specific_name]
                    assert class_general_name == self.GENERAL_CLASSES[gt_label]
                    filename = osp.join(subset_, f'{subset_}_image_{i}.byte')
                    info = {
                        'img_prefix': self.data_prefix,
                        'img_info': {
                            'filename': filename
                        },
                        'gt_label': np.array(gt_label, dtype=np.int64),
                    }
                    if not is_unzip_file:
                        info['img_bytes'] = img_bytes[i]
                    data_infos.append(info)
        return data_infos
