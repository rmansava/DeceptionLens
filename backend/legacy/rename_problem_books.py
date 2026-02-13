"""
Rename books with problematic characters in their names.
Renames in both:
- T:/disk-features/books (source features)
- D:/books/pdf-images (source images)
"""

import os
import re
import shutil

FEATURES_DIR = "T:/disk-features/books"
IMAGES_DIR = "D:/books/pdf-images"

# Books that failed to index (from comparison)
PROBLEM_BOOKS = """
1988 Olympics the complete guide
Cold War radio The dangerous history of American broadcasting in Europe 1950 1989
Comics between the panels - Steve Duin
Creatures of Darkness
Cricket On This Day History Facts and Figures from Every Day
Crime Comics The Illustrated History
Crime Fighting Heroes of Television Over 10000 Facts - Vincent Terrace
Crime Films Genres in American Cinema
Crime movies - Clarens Carlos
Crime movies from Griffith to The Godfather and beyond - Clarens Carlos
Crime Watching Investigating Real Crime TV
Critical Survey of Mystery and Detective Fiction
Croquet -  Ormerod
Croquet - 1988
Croquet - 1995
Croquet and How to Play It
Croquet including games for the garden and the Croquet - Gaunt Don
Croquet its history strategy rules and records
Croquet the gentle but wicket game
Croquet the skills of the game
Croquet the sport
Crosby Stills and Nash
Cross Your Fingers Spit in Your Hat Superstitions and
Crossroads How the Blues Shaped Rock n Roll (and Rock
Crossroads to the cinema
Crumbology the works of R Crumb 1981 to 1994
Cult baseball players the greats the flakes the weird
Cult British TV comedy
Cult Films - Allan Havis
Cult Horror Films_ From Attack of the 50 Foot Woman to - Welch Everman
Cult Media Repackaged Rereleased and Restored
Cult Movies in Sixty Seconds The Best Movies in the World in Less than a Minute
Cult of a Dark Hero
Cult Science Fiction Films From the Amazing Colossal Man to
Cult TV a viewers guide to the shows America cant live
Cult TV Heroines Angels Aliens and Amazons
Cult TV Mans ultimate modeling guide to classic scifi
Cult TV The Essential Critical Guide
Cultural Criticism 1969 1990 From Architectural Damages to
Cupboards of Curiosity Women Recollection and Film
Curly Lambeau building the Green Bay Packers
Cyberpunk Women Feminism and Science Fiction A Critical Study
Cyborgs Santa Claus and Satan Science Fiction Fantasy
Cyclopedia Of World Authors - Frank N Magill
Cynthia Lucia Framing Female Lawyers
Damn Right IVe Got The Blues
Dance Me a Song
Dandy and Beano  fifty years of fun
Dark city dames the wicked women of film noir
Daytime divas the dish on dozens of daytime TVs great - Kathleen Tracy
Daytime Television Game Shows and the Celebration of - Morris B Holbrook
DC Comics Encyclopedia New Edition
Dead Famous An Unexpected History of Celebrity from Bronze
Dead Reckonings The Life And Times Of The Grateful Dead
Denzel Washington his films and career
Detecting Women Gender and the Hollywood Detective Film
Detectionary a biographical dictionary of leading - Otto Penzler
Detroit Sluggers The First 75 Years
Encyclopedia of International Games
Encyclopedia of Video Games [3 volumes] The Culture, Technology, and Art of Gaming
Great lovers of the movies -- [by] Jane Mercer
Great moments in golf -- [by] Nevin H_ Gibson
Harmony Illustrated Encyclopedia of Rock, Fifth Edition -- [edited by] Mike Clifford
Have Gun―Will Travel (TV Milestones) -- Gaylyn Studlar
I Lost It at the Video Store [Expanded Edition]_ A -- Tom Roston
Ian Scott (auth.) - From Pinewood to Hollywood_ British Filmmakers in American Cinema, 1910-1969 (2010, Palgrave Macmillan UK) [10.1007_978-0-230-28973-4] - libgen.li
Index to characters in the performing arts_ 2,[2]_ Operas, -- Harold S_ Sharp 2
Individual sports for men -- [by] John H_ Shaw
Italian Sword and Sandal Films, 1908Ð1990 -- Roy Kinnard
ivan tors daktari night of terror [ Big Little Book]
Jack Hunter - House of Horror [Old Edition]_ The Complete History of Hammer Films-The Tears Corporation_Creation (1994)
James Dean The Mutant King A Biography
Kangaroos & other creatures from Down Under based on the -- Jackson, Donald Dale, 1935- -- 1977 -- [New York] Time-Life Films -- 9780913948170 -- 618d26ac833e7bcf01ee491bea2bb391 -- Annas Archive
Kompare, Derek - CSI (Kompare_CSI) __ Appendix_ CSI Episode Guide, 2000-9 (2010, Wiley-Blackwell) [10.1002_9781444328028.app1] - libgen.li
Linda Hall Dolores del Río Beauty in Light and Shade Stanford University Press (2013)
Luker J.H. Specialised illustrated Catalogue. Vol. One. World matchbox label series, foreign made, 1920 1972″
Lyle Price Guide_ Dolls and Toys -- [editor, Tony Curtis]
Manhattan dating game an unofficial and unauthorised guide -- Smith, Jim, 1978- -- [Updated ed.]., London, 2004 -- London Virgin -- 9780753509258 -- 7451bcef7d4626f6fdc84e59ec023886 -- Annas Archive
Mass news_ practices, controversies, and alternatives_ -- [Edited by] David J_ LeRoy
Michael Jackson_ All the Songs_ The Story Behind Every Track -- François Allard
Movie magic -- Cross, Robin -- 1994 -- Hemel Hempstead, Herts. [England] Simon & Schuster -- 9780750015455 -- 338067c9a49a8d906eeaeeb698d10176 -- Annas Archive
Official encyclopedia of tennis -- edited by the staff of the U_S_L_T_A -- [1st ed_]
Olympics olympics & paralympics -- Pegasus Team -- 2012 -- [Place of publication not identified] B Jain Publishers Pvt -- 9788131919156 -- 857541ca717ff5545fdb216f77b9baf7 -- Annas Archive
On the History of Rock Music -- Yvetta Kajanová -- 2014 -- Peter Lang GmbH -- 9783631655566 -- befd76dc1fea813820a71c2512ab40b7 -- Annas Archive
Once was enough celebrities (and others) who appeared a -- Brode, Douglas, 1943- -- 1. [Dr.]., Secaucus, 1997 -- Secaucus, N.J. Carol Pub. -- 9780806517353 -- b987dd4a94d31162cb1cf4e298de1543 -- Annas Archive
Popular TV shows of the 90s Friends, Seinfeld, ER, -- Reese, Jenny -- 2011 -- [Place of publication not identified] Six Degrees Books -- 9781170680827 -- 4284da947003dc51c9373a3dbd121538 -- Annas Archive
Purnells New Encyclopedia of Association Football -- [edited by] Norman S_ Barrett
Radio and TV Boy Potter 1972
Rock On, Volume II _ The Illustrated Encyclopedia of Rock N -- [by] Norm N_ Nite
Roman Base Metal Coins - A Price Guide _ Каталог-ценник -- Richard J_ Plant
Singing Out An Oral History of Americas Folk Music Revivals
Spencer tracy tragic idol -- Davidson, Bill -- 1992 -- [Place of publication not identified] Kensington Pub Corp -- 9780821737385 -- 97e4ab7ef2ad77ff3307ebeabb310779 -- Annas Archive
Television Game Show Hosts  Biographies of 32 Stars
The  Hamlyn history of the movies -- [by] Mary Davies
The Big Show -- Keith Olbermann, Dan Patrick -- New York, ©1997
The complete encyclopedia of popular music and jazz 1900 - -- [by] Roger D_ Kinkle
The complete encyclopedia of popular music and jazz 1900 - -- [by] Roger D_ Kinkle 2
The detective in film -- [by] William K_ Everson
The Elvis encyclopedia the complete and definitive -- Stanley, David, 1955-; Coffey, Frank -- Repr., [Emmaus, Pa], 2002 -- North Dighton, Mass. JG -- 9781572153196 -- f362e2b67c38ad735d6624acac241a14 -- Annas Archive
The fabulous fantasy films -- Rovin, Jeff -- 1977 -- South Brunswick [N.J.] A.S. Barnes -- 9780498018039 -- 11f72844341b83f69203cb3b03206a34 -- Annas Archive
The films of Frank Sinatra -- [by] Gene Ringgold
The films of Woody Allen -- Brode, Douglas, 1943- -- 1991 -- Secaucus, NJ Carol Publishing ; [London] [Virgin Books] -- 9780863695780 -- 6d8aed9bae3baa7f2b1ff28673dce14e -- Annas Archive
The golden age of B movies -- McClelland, Doug -- 1981 -- [New York, N.Y.] Bonanza Books Distributed by Crown -- 9780517349229 -- 575fa474a4784b1f3b9a6ef125b43680 -- Annas Archive
The gospel according to Madison Avenue -- [by] Ray Hutchinson
The great radio comedians_ -- Jim Harmon -- [1st ed_], Garden City, N_Y, New York State, 1970
The great television series -- Rovin, Jeff -- (4. print)., South Brunswick [usw.], 1977 -- South Brunswick [N.J.] A. S. Barnes -- 9780498019616 -- 69f430169732eed807f7a3764d0404df -- Annas Archive
The Harmony illustrated encyclopedia of rock -- consultant, Mike Clifford; authors, Pete Frame ... [et al.] -- 1988 -- New York Harmony Books -- 9780517571644 -- 88fd60fc0d9043f20e8236702b9559a4 -- Annas Archive
The Horror Handbook Slasher Movies (2014) [ZomBiRG]
The Horror Show Guide (The Ultimate Frightfest of Movies) [ZomBiRG]
The Internet Guide for the Movie Addict[Team Nanban][TPB]
The movies [an illustrated history of the silver screen] -- Shiach, Don -- 2000 -- London Hermes House -- 9781840385533 -- 88e66c184a85a57a5c948223fa1ff51e -- Annas Archive
The NME[New Musical Express] Rock`n`Roll years -- David Heslam
The Official NFL encyclopedia_ sucks -- [edited by] Beau Riffenburgh
The Olympics a history of the games -- Johnson, William O., 1931-2012 -- [Birmingham, Ala.], 1992 -- [Birmingham, Ala.] Oxmoor House -- 9780848711153 -- ff7dd11303f2925917c635c4470c7af2 -- Annas Archive
The Original Folk and Fairy Tales of the Brothers Grimm the complete first edition
The Peoples Almanac [no_] 3 -- David Wallechinsky
The Peoples Almanac presents The book of lists -- [compiled] by David Wallechinsky
The Robert Shaw Reader
The tennis book -- [by] Larry Lorimer; with ill_ by Elizabeth Roger
The Unauthorized X-Cyclopedia_ The Definitive Reference   James Hatfield  ©1997
The Variety guide to film festivals_ the ultimate insiders -- [edited by] Steven Gaydos
The Virgin Encyclopedia of Reggae (Virgin Encyclopedias of -- [editor] Colin Larkin
To Be Continued___a Complete Guide to over Two Hundred -- [by] Ken Weiss
Top 1000 Singles 1955 - 1990 Billboard See 183081 -- [compiled by Joel Whitburn]
Wally Raccoons farmyard Olympics. Team sports -- Hope, Leela -- 2016 -- [United States ] [CreateSpace Independent Publishing -- 9789657736487 -- 5d5f84a13c0aae5f6698accb20ae6e3f -- Annas Archive
Welcome to the ancient Olympics! [ancient Greek Olympics] -- Bingham, Jane -- 2008 -- Oxford Raintree -- 9781406207644 -- 717226b080d0e6d12e95dffa504864b4 -- Annas Archive
World History Encyclopedia [21 volumes]
""".strip().split('\n')


def sanitize_name(name):
    """Create a clean filename from a book name."""
    # Remove metadata patterns like "-- [by] Author" or "-- Author -- Year"
    name = re.sub(r'\s*--.*$', '', name)

    # Remove brackets and their contents
    name = re.sub(r'\[.*?\]', '', name)
    name = re.sub(r'\(.*?\)', '', name)

    # Remove special characters
    name = re.sub(r'[_©―`″]', ' ', name)
    name = re.sub(r'[,;:]', '', name)
    name = re.sub(r'&', 'and', name)

    # Clean up whitespace
    name = re.sub(r'\s+', ' ', name).strip()

    # Truncate if too long
    if len(name) > 100:
        name = name[:100].rsplit(' ', 1)[0]

    return name


def main():
    print("=" * 70)
    print("  RENAME PROBLEM BOOKS")
    print("=" * 70)
    print()

    renames = []

    for book in PROBLEM_BOOKS:
        book = book.strip()
        if not book:
            continue

        new_name = sanitize_name(book)

        if new_name != book:
            renames.append((book, new_name))

    print(f"  Found {len(renames)} books to rename")
    print()

    # Check for duplicates in new names
    new_names = [r[1] for r in renames]
    duplicates = set([n for n in new_names if new_names.count(n) > 1])
    if duplicates:
        print(f"  WARNING: {len(duplicates)} duplicate new names, adding suffixes...")
        # Add suffix to duplicates
        name_counts = {}
        for i, (old, new) in enumerate(renames):
            if new in duplicates:
                name_counts[new] = name_counts.get(new, 0) + 1
                if name_counts[new] > 1:
                    renames[i] = (old, f"{new} {name_counts[new]}")

    print()
    print("-" * 70)
    print("  Preview (first 10):")
    print("-" * 70)
    for old, new in renames[:10]:
        print(f"  {old[:50]}...")
        print(f"    -> {new}")
        print()

    print("-" * 70)
    input("  Press Enter to rename, or Ctrl+C to cancel...")
    print()

    renamed = 0
    errors = []

    for old, new in renames:
        # Rename in features dir
        features_old = os.path.join(FEATURES_DIR, old)
        features_new = os.path.join(FEATURES_DIR, new)

        # Rename in images dir
        images_old = os.path.join(IMAGES_DIR, old)
        images_new = os.path.join(IMAGES_DIR, new)

        try:
            # Check if source exists
            has_features = os.path.exists(features_old)
            has_images = os.path.exists(images_old)

            if not has_features and not has_images:
                print(f"  SKIP: {old} (not found)")
                continue

            # Check if target already exists
            if os.path.exists(features_new) or os.path.exists(images_new):
                print(f"  SKIP: {new} (target exists)")
                continue

            # Rename
            if has_features:
                os.rename(features_old, features_new)
            if has_images:
                os.rename(images_old, images_new)

            renamed += 1
            print(f"  OK: {old[:40]}... -> {new[:40]}...")

        except Exception as e:
            errors.append((old, str(e)))
            print(f"  ERROR: {old}: {e}")

    print()
    print("=" * 70)
    print(f"  DONE! Renamed {renamed} books")
    if errors:
        print(f"  Errors: {len(errors)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
