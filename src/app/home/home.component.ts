import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { ElectronService } from 'app/core/services';
import { of } from 'rxjs';


interface Tag {
  name: string,
  parent: string,
}
interface Word {
  name: string,
  tag: string,
}

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

  constructor(
    private router: Router,
    private electronService: ElectronService
  ) {

  }
  ngOnInit(): void {

  }

  public newWord: string = "";
  public currentTag: string = "";
  public wordList: Array<Word> = [];
  private wordSet: Set<string> = new Set<string>();
  public tagMap: Map<string, Tag> = new Map();
  public addWord(word: string, tag: string) {
    if (this.wordSet.has(word)) {
      return;
    }
    this.wordSet.add(word);
    this.wordList.push({
      name: word,
      tag: tag,
    });
  }
  public addTag(tag: string, parent?: string) {
    if (this.tagMap.has(tag)) {
      return;
    }
    if (!parent) {
      parent = "";
    }
    this.tagMap.set(tag, {
      name: tag,
      parent: parent,
    });
    this.tagList.push(tag);
  }
  public tagList: Array<string> = [];
  public keyEnter() {
    console.log(this.currentTag)
    this.addWord(this.newWord, this.currentTag);
    this.newWord = "";
  }
  public newTag: string = "";
  public tagKeyEnter() {
    this.addTag(this.newTag);
    this.newTag = ""
  }
  public save() {
    this.electronService.save({
      'words': this.wordList,
      'tags': Array.from(this.tagMap.entries()),
    });
  }
  public load() {
    let d = this.electronService.load();
    if ('words' in d) {
      this.wordList = d['words'];
      this.refreshWordSet();
    }
    if ('tags' in d) {
      this.tagMap = new Map(d['tags']);
      this.refreshTagSet();
    }
  }
  refreshWordSet() {
    this.wordSet = new Set<string>();
    for (let word of this.wordList) {
      this.wordSet.add(word.name);
    }
  }
  refreshTagSet() {
    this.tagList = Array.from(this.tagMap.keys());
  }
}
