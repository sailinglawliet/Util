#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 16:04:19 2016
A TrieTree with node contains leaf's value and each node's depth

@Author : Adryan
@MailTo : lihongyuan@hzgit.com
@Version: 1.0
"""


TRIE_ELEMENT = "0123456789.abcdefghijklmnopqrstuvwxyz"

class TreeNode():
    def __init__(self):
        self.depth = 1
        self.value = None
        self.t = {}
        


class TrieTree():
    def __init__(self):
        self.root = TreeNode()
    
    
    def add(self, word):
        node = self.root
        index = 0
        for char in word:
            index += 1
            if char in node.t:
                node = node.t[char]
            else:
                node.t[char] = TreeNode()
                node = node.t[char]
                node.depth = index
        node.value = word
    
    
    def search(self, word):
        node = self.root
        for char in word:
            if char in node.t:
                node = node.t[char]
            else:
                return False
        if node.value == word:
                return True
        else:
            return False

            
    def find(self, prefix):
        if self.search(prefix):
            return [prefix]
        else:
            r = []
            node = self.root
            for char in prefix:
                if char in node.t:
                    node = node.t[char]
                else:
                    return r
            r = self.display(node, r)
            return r
    
                    
    def display(self, node, r):
        if node.value != None:
            #print node.value
            r.append(node.value)
        for char in TRIE_ELEMENT:
            if char in node.t:
                self.display(node.t[char], r)
        return r
            
                

def main():
    trie = TrieTree()
    trie.add('beckman')
    trie.add('belfield')
    trie.add('angelo')
    print trie.find('be')
    

if __name__=='__main__':
    main()