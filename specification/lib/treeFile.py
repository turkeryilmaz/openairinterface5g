from typing import List
import re
class YangTreeItem:

    def __init__(self, name:str, access:str, listKey:str=None, mandatory=False, type:str=None ) -> None:
        self.children:List[YangTreeItem]=[]
        self.name = name
        self.access = access
        self.listKey = listKey
        self.mandatory = mandatory
        self.type = type

    def addChild(self, item):
        self.children.append(item)

    def __str__(self) -> str:
        s=f'YangTreeItem[name={self.name}, access={self.access}]'

    @staticmethod
    def fromMatch(match:re.Match):
        indent = len(match.group(1))
        treeLevel = 0 if indent==2 else int((indent-2)/3)
        access = match.group(2)
        name = match.group(3)
        listKey = match.group(6)
        hasStar = match.group(4)
        itemType = "list" if listKey is not None else "container" if match.group(7) is None else match.group(7)
        item = YangTreeItem(name, access, listKey=listKey, mandatory=hasStar, type=itemType)
        return treeLevel, item
 
class YangTreeModule:

    def __init__(self, name:str) -> None:
        self.name = name
        self.elems:List[YangTreeItem]=[]

    def addRootElem(self, item:YangTreeItem) -> None:
        self.elems.append(item)
    
    def __str__(self) -> str:
        s=f'YangTreeModule[name={self.name}, elems=\n'
        for elem in self. elems:
            s+=f'{elem}\n'
        s+="]"
        return s
   
class TreeFile:

    REGEX_MODULE = r"^module:\ (.*)$"
    #REGEX_ITEMS = r"^([\ |]+)\+--([rwo]+)\ ([^*\ \?]+)([\*\?]?)(\ +(\[([^\]]+)\])?([^\ ]+)?)?$"
    REGEX_ITEMS = r"^([\ |]+)\+--([rwo:]+)\ ([^*\ \?]+)([\*\?]?)(\ \[([^\]]+)\])?"

    def __init__(self, filename:str) -> None:
        self.filename = filename
        

    
    def parse(self)->List[YangTreeModule]:

        modules = []
        curModule:YangTreeModule=None
        curItems=dict()
        curItem=None
        with open(self.filename) as fp:
            line = fp.readline()
            while line is not None:
                print(f'parsing line {line}')
                matches = re.finditer(TreeFile.REGEX_MODULE, line)
                match = next(matches, None)
                # found new module
                if match is not None:
                    if curModule is not None:
                        modules.append(curModule)
                    curModule = YangTreeModule(match.group(1))
                else:
                    matches = re.finditer(TreeFile.REGEX_ITEMS, line)
                    match = next(matches, None)
                    # found new container|list|...
                    if match is not None:
                        treeLevel, item = YangTreeItem.fromMatch(match)
                        if treeLevel == 0:
                            curModule.addRootElem(item)
                        else:
                            curItems[treeLevel-1].addChild(item)
                        curItems[treeLevel]=item
                        curItem=item
                    else:
                        

                line = fp.readline()


        for module in modules:
            print(module)


file = TreeFile('specification/xml/gnodeb.tree')
file.parse()